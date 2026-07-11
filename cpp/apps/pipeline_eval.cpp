#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "benchmarks/metrics.h"
#include "deepscores/dataset.h"
#include "deepscores/io.h"
#include "pipeline_common.h"
#include "symbols/evaluation.h"
#include "symbols/splitter.h"

namespace
{
	using json = nlohmann::ordered_json;

	using clock_type = std::chrono::steady_clock;

	double elapsed_ms(clock_type::time_point start, clock_type::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - start).count();
	}

	// Attach wall-clock timings (ms) covering the whole per-page pipeline.
	// discover_ms: crop discovery incl. image decode / I/O.
	// eval_ms: inference + reconstruction + metrics.
	// elapsed_ms: total per-page wall time (discover + eval).
	void set_timings(json& page, double discover_ms, double eval_ms)
	{
		page["discover_ms"] = discover_ms;
		page["eval_ms"] = eval_ms;
		page["elapsed_ms"] = discover_ms + eval_ms;
	}

	const std::vector<std::string> CLASSES = {
		"noteheadBlackOnLine",
		"noteheadBlackInSpace",
		"stem",
		"beam",
	};

	const std::vector<std::string> MASK_KINDS = {
		"notehead",
		"stem",
		"beam",
		"unknown",
	};

	struct ManifestEntry
	{
		std::string band;
		std::string image_id;
		int python_n_crops;
	};

	std::vector<ManifestEntry>
	read_page_ids_file(const std::filesystem::path& path)
	{
		std::ifstream f(path);
		if (!f)
			throw std::runtime_error("failed to open " + path.string());
		std::vector<ManifestEntry> entries;
		std::string line;
		std::getline(f, line); // header
		while (std::getline(f, line))
		{
			if (line.empty())
				continue;
			std::istringstream ss(line);
			std::string band, image_id, n_crops_str;
			std::getline(ss, band, '\t');
			std::getline(ss, image_id, '\t');
			std::getline(ss, n_crops_str, '\t');
			entries.push_back({ band, image_id, std::stoi(n_crops_str) });
		}
		return entries;
	}

	json mask_scores_to_json(
		const std::map<std::string, std::vector<omr::MaskScores>>&
			mask_scores_per_kind)
	{
		json out = json::object();
		for (const auto& kind : MASK_KINDS)
		{
			auto it = mask_scores_per_kind.find(kind);
			if (it == mask_scores_per_kind.end())
				continue;
			const auto& scores = it->second;
			double p = 0, r = 0, f = 0, i = 0;
			for (const auto& s : scores)
			{
				p += s.precision;
				r += s.recall;
				f += s.f1;
				i += s.iou;
			}
			auto avg = [&](double total) {
				return scores.empty()
					? 0.0
					: total / static_cast<double>(scores.size());
			};
			out[kind] = {
				{ "precision", avg(p) },
				{ "recall", avg(r) },
				{ "f1", avg(f) },
				{ "iou", avg(i) },
			};
		}
		return out;
	}

	json confusion_to_json(const std::vector<std::vector<int>>& matrix)
	{
		json out = json::array();
		for (const auto& row : matrix)
		{
			json json_row = json::array();
			for (int v : row)
				json_row.push_back(v);
			out.push_back(json_row);
		}
		return out;
	}

	json per_class_ap_to_json(
		const std::unordered_map<std::string, double>& per_class_ap)
	{
		json out = json::object();
		for (const auto& cls : CLASSES)
		{
			auto it = per_class_ap.find(cls);
			if (it != per_class_ap.end())
				out[cls] = it->second;
		}
		return out;
	}

	double safe_mean(const std::vector<double>& values)
	{
		if (values.empty())
			return 0.0;
		double sum = 0.0;
		for (double v : values)
			sum += v;
		return sum / static_cast<double>(values.size());
	}

	json crop_diagnostics_to_json(const omr::CropDiagnostics& diagnostics)
	{
		return {
			{ "candidate_target_annotations",
			  diagnostics.candidate_target_annotations },
			{ "components_considered", diagnostics.components_considered },
			{ "rejected_no_supported_notehead",
			  diagnostics.rejected_no_supported_notehead },
			{ "rejected_stacked_head", diagnostics.rejected_stacked_head },
			{ "rejected_semantic_union_empty",
			  diagnostics.rejected_semantic_union_empty },
			{ "rejected_no_ink_touched", diagnostics.rejected_no_ink_touched },
			{ "yielded_crops", diagnostics.yielded_crops },
			{ "images_without_noteheads",
			  diagnostics.images_without_noteheads },
			{ "balances", diagnostics.balances() },
		};
	}

	void dump_page_artifacts(
		const omr::DeepScoresDataset& dataset, const std::string& image_id,
		const std::vector<omr::CropSample>& samples,
		const std::vector<omr::pipeline::CropResult>& crop_results,
		const std::filesystem::path& viz_dir)
	{
		auto [original_mask, staffless_mask, _] = dataset.load_page(image_id);
		(void)original_mask;
		cv::Mat3b canvas(staffless_mask.mat().size(), cv::Vec3b(255, 255, 255));
		canvas.setTo(cv::Vec3b(184, 184, 184), staffless_mask.mat());
		const cv::Vec3b notehead_color(140, 26, 217);
		const cv::Vec3b stem_color(230, 115, 26);
		const cv::Vec3b beam_color(13, 140, 255);
		const cv::Vec3b unknown_color(31, 31, 230);
		const cv::Scalar link_color(0, 255, 0);
		const cv::Scalar accepted_color(0, 255, 0);
		const cv::Scalar uncertain_color(0, 215, 255);

		for (std::size_t i = 0; i < samples.size() && i < crop_results.size();
			 ++i)
		{
			const auto& sample = samples[i];
			const auto& split = crop_results[i].split;
			int ox = sample.origin.first;
			int oy = sample.origin.second;

			for (const auto& prim : split.primitives)
			{
				cv::Vec3b color;
				switch (prim.kind)
				{
				case omr::PrimitiveKind::NOTEHEAD:
					color = notehead_color;
					break;
				case omr::PrimitiveKind::STEM:
					color = stem_color;
					break;
				case omr::PrimitiveKind::BEAM:
					color = beam_color;
					break;
				case omr::PrimitiveKind::UNKNOWN:
					color = unknown_color;
					break;
				}
				for (int y = 0; y < prim.mask.rows(); ++y)
					for (int x = 0; x < prim.mask.cols(); ++x)
						if (prim.mask.at(y, x) && oy + y >= 0
							&& oy + y < canvas.rows && ox + x >= 0
							&& ox + x < canvas.cols)
							canvas(oy + y, ox + x) = color;
			}

			auto center = [&](int idx) -> std::pair<double, double> {
				const auto& p = split.primitives[idx];
				return { (p.bbox.x0 + p.bbox.x1) / 2.0 + ox,
						 (p.bbox.y0 + p.bbox.y1) / 2.0 + oy };
			};
			auto add_link = [&](int s, int t) {
				auto [x0, y0] = center(s);
				auto [x1, y1] = center(t);
				cv::line(canvas, cv::Point(cvRound(x0), cvRound(y0)),
						 cv::Point(cvRound(x1), cvRound(y1)), link_color, 1,
						 cv::LINE_AA);
			};
			for (const auto& [s, t] : split.head_to_stem)
				add_link(s, t);
			for (const auto& [s, t] : split.stem_to_beam)
				add_link(s, t);

			if (split.status != omr::SplitStatus::REJECTED)
			{
				const cv::Scalar& box_color =
					split.status == omr::SplitStatus::ACCEPTED
					? accepted_color
					: uncertain_color;
				cv::rectangle(canvas,
							  cv::Rect(ox, oy, sample.staffless_mask.cols(),
									   sample.staffless_mask.rows()),
							  box_color, 1);
			}
		}
		const int scale = std::max(1, canvas.cols / 1800);
		cv::rectangle(canvas,
					  cv::Rect(12 * scale, 12 * scale, 430 * scale, 76 * scale),
					  cv::Scalar(255, 255, 255), cv::FILLED);
		cv::putText(canvas, "notehead", cv::Point(24 * scale, 42 * scale),
					cv::FONT_HERSHEY_SIMPLEX, 0.55 * scale,
					cv::Scalar(notehead_color), scale, cv::LINE_AA);
		cv::putText(canvas, "stem", cv::Point(155 * scale, 42 * scale),
					cv::FONT_HERSHEY_SIMPLEX, 0.55 * scale,
					cv::Scalar(stem_color), scale, cv::LINE_AA);
		cv::putText(canvas, "beam", cv::Point(245 * scale, 42 * scale),
					cv::FONT_HERSHEY_SIMPLEX, 0.55 * scale,
					cv::Scalar(beam_color), scale, cv::LINE_AA);
		cv::putText(canvas, "unknown", cv::Point(310 * scale, 42 * scale),
					cv::FONT_HERSHEY_SIMPLEX, 0.55 * scale,
					cv::Scalar(unknown_color), scale, cv::LINE_AA);
		cv::putText(canvas, "association", cv::Point(24 * scale, 72 * scale),
					cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, link_color, scale,
					cv::LINE_AA);
		std::filesystem::create_directories(viz_dir);
		const auto output = viz_dir / (image_id + ".png");
		if (!cv::imwrite(output.string(), canvas))
			throw std::runtime_error("failed to write " + output.string());
	}

	json evaluate_page(
		const omr::DeepScoresDataset& dataset,
		const omr::PrimitiveSplitter& splitter, const std::string& image_id,
		const std::vector<omr::CropSample>& samples, double iou_threshold,
		const std::filesystem::path& data_dir, const std::string& band = "",
		const omr::CropDiagnostics* crop_diagnostics = nullptr,
		std::vector<omr::pipeline::CropResult>* out_crop_results = nullptr)
	{
		const omr::DatasetImage& image = dataset.images.at(image_id);
		const auto image_path = data_dir / "images" / image.filename;
		std::vector<double> staff_rows;
		if (std::filesystem::exists(image_path))
			staff_rows = omr::pipeline::recover_staff_rows(image_path);
		const double staff_line = omr::pipeline::compute_staff_line(samples);

		std::vector<omr::pipeline::CropResult> crop_results;
		std::vector<std::vector<double>> staff_lines_per_crop;
		std::map<std::string, std::vector<omr::MaskScores>>
			mask_scores_per_kind;

		for (const auto& sample : samples)
		{
			auto cr = omr::pipeline::process_crop(splitter, sample, staff_rows,
												  staff_line);
			staff_lines_per_crop.push_back(cr.system_lines);
			if (!sample.truth_masks.empty())
			{
				std::map<std::string, omr::Mask> truth_map(
					sample.truth_masks.begin(), sample.truth_masks.end());
				auto scores = omr::evaluate_split(cr.split, truth_map,
												  &sample.staffless_mask);
				for (auto& [kind, score] : scores)
					mask_scores_per_kind[kind].push_back(score);
			}
			crop_results.push_back(std::move(cr));
		}

		std::vector<omr::PredictedCropResult> crop_refs;
		crop_refs.reserve(samples.size());
		for (std::size_t i = 0; i < crop_results.size(); ++i)
		{
			crop_refs.push_back({ &crop_results[i].split,
								  &crop_results[i].graph, samples[i].origin });
		}

		omr::PageIO predicted = omr::build_predicted_page(
			image, crop_refs, staff_lines_per_crop, staff_line);
		// Ground truth is loaded only after inference and covers the full page.
		omr::PageIO truth = omr::derive_ground_truth(
			dataset, image_id, std::nullopt, nullptr, staff_rows);
		int covered = 0;
		int duplicate_covered = 0;
		std::unordered_set<int> useful_crops;
		for (const auto& annotation : truth.annotations)
		{
			const double cx = (annotation.bbox.x0 + annotation.bbox.x1) / 2.0;
			const double cy = (annotation.bbox.y0 + annotation.bbox.y1) / 2.0;
			int hits = 0;
			for (std::size_t i = 0; i < samples.size(); ++i)
			{
				const auto& sample = samples[i];
				const int x0 = sample.origin.first;
				const int y0 = sample.origin.second;
				if (cx >= x0 && cx < x0 + sample.staffless_mask.cols()
					&& cy >= y0 && cy < y0 + sample.staffless_mask.rows())
				{
					++hits;
					useful_crops.insert(static_cast<int>(i));
				}
			}
			covered += hits > 0;
			duplicate_covered += hits > 1;
		}

		auto matches = omr::match_pairs(predicted.annotations,
										truth.annotations, iou_threshold);
		auto [mAP, per_class_ap] = omr::mean_average_precision(
			predicted.annotations, truth.annotations, CLASSES, iou_threshold);
		auto [confusion, accuracy] = omr::confusion_matrix(matches, CLASSES);
		auto [edge_p, edge_r, edge_f1] =
			omr::edge_prf1(predicted.graph_edges, truth.graph_edges, matches);
		double ser = omr::symbol_error_rate(predicted.duration_sequence,
											truth.duration_sequence);
		double pitch_ser = omr::symbol_error_rate(predicted.pitch_sequence,
												  truth.pitch_sequence);

		json page;
		page["image_id"] = image_id;
		page["filename"] = image.filename;
		if (!band.empty())
			page["band"] = band;
		page["n_crops"] = static_cast<int>(samples.size());
		if (crop_diagnostics)
			page["crop_diagnostics"] =
				crop_diagnostics_to_json(*crop_diagnostics);
		page["mask_prf1"] = mask_scores_to_json(mask_scores_per_kind);
		page["mAP"] = mAP;
		page["per_class_ap"] = per_class_ap_to_json(per_class_ap);
		page["accuracy"] = accuracy;
		page["confusion_matrix"] = confusion_to_json(confusion);
		page["edge_precision"] = edge_p;
		page["edge_recall"] = edge_r;
		page["edge_f1"] = edge_f1;
		page["ser"] = ser;
		page["pitch_ser"] = pitch_ser;
		page["n_pred"] = static_cast<int>(predicted.annotations.size());
		page["n_gt"] = static_cast<int>(truth.annotations.size());
		page["coverage"] = truth.annotations.empty()
			? 0.0
			: static_cast<double>(covered) / truth.annotations.size();
		page["covered_gt"] = covered;
		page["duplicate_covered_gt"] = duplicate_covered;
		page["useful_crops"] = static_cast<int>(useful_crops.size());
		page["irrelevant_crops"] = static_cast<int>(samples.size())
			- static_cast<int>(useful_crops.size());
		page["pred_duration_sequence"] = predicted.duration_sequence;
		page["gt_duration_sequence"] = truth.duration_sequence;
		page["pred_pitch_sequence"] = predicted.pitch_sequence;
		page["gt_pitch_sequence"] = truth.pitch_sequence;
		if (out_crop_results)
			*out_crop_results = std::move(crop_results);
		return page;
	}

	json aggregate(const std::vector<json>& pages)
	{
		json summary = json::object();
		if (pages.empty())
			return summary;

		std::vector<int> n_crops_list;
		n_crops_list.reserve(pages.size());
		for (const auto& p : pages)
			n_crops_list.push_back(p["n_crops"].get<int>());
		int crop_total = 0, crop_min = INT_MAX, crop_max = 0;
		for (int v : n_crops_list)
		{
			crop_total += v;
			if (v < crop_min)
				crop_min = v;
			if (v > crop_max)
				crop_max = v;
		}
		double crop_mean =
			static_cast<double>(crop_total) / static_cast<double>(pages.size());
		std::vector<int> sorted_crops = n_crops_list;
		std::sort(sorted_crops.begin(), sorted_crops.end());
		double crop_median;
		if (sorted_crops.size() % 2 == 1)
			crop_median = sorted_crops[sorted_crops.size() / 2];
		else
			crop_median = (sorted_crops[sorted_crops.size() / 2 - 1]
						   + sorted_crops[sorted_crops.size() / 2])
				/ 2.0;
		json crop_stats;
		crop_stats["total"] = crop_total;
		crop_stats["mean"] = crop_mean;
		crop_stats["median"] = crop_median;
		crop_stats["min"] = pages.size() > 0 ? crop_min : 0;
		crop_stats["max"] = crop_max;
		json zero_crop_ids = json::array();
		json per_page_crops = json::object();
		for (const auto& p : pages)
		{
			const std::string image_id = p["image_id"].get<std::string>();
			per_page_crops[image_id] = p["n_crops"].get<int>();
			if (p["n_crops"].get<int>() == 0)
				zero_crop_ids.push_back(image_id);
		}
		crop_stats["zero_crop_pages"] = zero_crop_ids;
		crop_stats["per_page"] = per_page_crops;
		std::map<std::string, std::vector<int>> by_band;
		for (const auto& p : pages)
		{
			std::string b =
				p.contains("band") ? p["band"].get<std::string>() : "unknown";
			by_band[b].push_back(p["n_crops"].get<int>());
		}
		json band_stats = json::object();
		for (const auto& [b, v] : by_band)
		{
			int bt = 0;
			for (int x : v)
				bt += x;
			band_stats[b] = {
				{ "total", bt },
				{ "mean",
				  v.empty() ? 0.0
							: static_cast<double>(bt)
						  / static_cast<double>(v.size()) },
			};
		}
		crop_stats["by_band"] = band_stats;

		omr::CropDiagnostics diagnostics_total;
		std::map<std::string, omr::CropDiagnostics> diagnostics_by_band;
		bool has_diagnostics = false;
		for (const auto& page : pages)
		{
			if (!page.contains("crop_diagnostics")
				|| page["crop_diagnostics"].is_null())
				continue;
			const auto& value = page["crop_diagnostics"];
			omr::CropDiagnostics diagnostics;
			diagnostics.candidate_target_annotations =
				value.value("candidate_target_annotations", 0);
			diagnostics.components_considered =
				value.value("components_considered", 0);
			diagnostics.rejected_no_supported_notehead =
				value.value("rejected_no_supported_notehead", 0);
			diagnostics.rejected_stacked_head =
				value.value("rejected_stacked_head", 0);
			diagnostics.rejected_semantic_union_empty =
				value.value("rejected_semantic_union_empty", 0);
			diagnostics.rejected_no_ink_touched =
				value.value("rejected_no_ink_touched", 0);
			diagnostics.yielded_crops = value.value("yielded_crops", 0);
			diagnostics.images_without_noteheads =
				value.value("images_without_noteheads", 0);
			diagnostics_total += diagnostics;
			const std::string band = page.contains("band")
				? page["band"].get<std::string>()
				: "unknown";
			diagnostics_by_band[band] += diagnostics;
			has_diagnostics = true;
		}

		const std::vector<std::string> keys_float = {
			"mAP",	   "accuracy", "edge_precision", "edge_recall",
			"edge_f1", "ser",	   "pitch_ser",
		};
		for (const auto& key : keys_float)
		{
			std::vector<double> values;
			for (const auto& p : pages)
				values.push_back(p[key].get<double>());
			summary[key] = safe_mean(values);
		}

		std::set<std::string> mask_kinds;
		for (const auto& p : pages)
			for (const auto& [k, _] : p["mask_prf1"].items())
				mask_kinds.insert(k);
		json mask_means = json::object();
		for (const auto& kind : mask_kinds)
		{
			std::map<std::string, std::vector<double>> per_field;
			for (const auto& p : pages)
			{
				auto it = p["mask_prf1"].find(kind);
				if (it == p["mask_prf1"].end())
					continue;
				for (const auto& f : { "precision", "recall", "f1", "iou" })
					per_field[f].push_back(it->at(f).get<double>());
			}
			mask_means[kind] = {
				{ "precision", safe_mean(per_field["precision"]) },
				{ "recall", safe_mean(per_field["recall"]) },
				{ "f1", safe_mean(per_field["f1"]) },
				{ "iou", safe_mean(per_field["iou"]) },
			};
		}
		summary["n_pages"] = static_cast<int>(pages.size());
		summary["mask_prf1"] = mask_means;

		json reordered = json::object();
		reordered["n_pages"] = summary["n_pages"];
		reordered["crop_stats"] = crop_stats;
		double coverage_sum = 0.0;
		int useful_crops = 0;
		int irrelevant_crops = 0;
		int duplicate_covered_gt = 0;
		for (const auto& page : pages)
		{
			coverage_sum += page.value("coverage", 0.0);
			useful_crops += page.value("useful_crops", 0);
			irrelevant_crops += page.value("irrelevant_crops", 0);
			duplicate_covered_gt += page.value("duplicate_covered_gt", 0);
		}
		reordered["coverage"] = pages.empty()
			? 0.0
			: coverage_sum / static_cast<double>(pages.size());
		reordered["useful_crops"] = useful_crops;
		reordered["irrelevant_crops"] = irrelevant_crops;
		reordered["duplicate_covered_gt"] = duplicate_covered_gt;
		if (has_diagnostics)
		{
			reordered["crop_diagnostics_totals"] =
				crop_diagnostics_to_json(diagnostics_total);
			json diagnostics_band_json = json::object();
			for (const auto& [band, diagnostics] : diagnostics_by_band)
				diagnostics_band_json[band] =
					crop_diagnostics_to_json(diagnostics);
			reordered["crop_diagnostics_by_band"] = diagnostics_band_json;
		}
		reordered["mask_prf1"] = mask_means;
		for (const auto& key : keys_float)
			reordered[key] = summary[key];

		std::map<std::string, std::vector<double>> class_aps;
		for (const auto& p : pages)
			for (const auto& [c, ap] : p["per_class_ap"].items())
				class_aps[c].push_back(ap.get<double>());
		json per_class_mean = json::object();
		for (const auto& cls : CLASSES)
		{
			auto it = class_aps.find(cls);
			if (it != class_aps.end() && !it->second.empty())
				per_class_mean[cls] = safe_mean(it->second);
		}
		reordered["per_class_ap"] = per_class_mean;

		const int n = static_cast<int>(CLASSES.size());
		std::vector<std::vector<int>> confusion_total(n,
													  std::vector<int>(n, 0));
		for (const auto& p : pages)
		{
			const auto& cm = p["confusion_matrix"];
			for (int i = 0; i < n; ++i)
				for (int j = 0; j < n; ++j)
					confusion_total[i][j] += cm[i][j].get<int>();
		}
		reordered["confusion_matrix"] = confusion_to_json(confusion_total);

		return reordered;
	}

	void print_row(const json& page, int width)
	{
		double mask_f1 = 0.0;
		if (page["mask_prf1"].contains("notehead"))
			mask_f1 = page["mask_prf1"]["notehead"]["f1"].get<double>();
		std::printf("%*s %4d %5d %5d %8.4f %8.4f %8.4f %8.4f %8.4f\n",
					width / 2, page["image_id"].get<std::string>().c_str(),
					page["n_crops"].get<int>(), page["n_pred"].get<int>(),
					page["n_gt"].get<int>(), mask_f1, page["mAP"].get<double>(),
					page["accuracy"].get<double>(),
					page["edge_f1"].get<double>(), page["ser"].get<double>());
	}

	void print_summary(const json& summary, double iou_threshold)
	{
		int n_pages = 0;
		if (summary.contains("n_pages"))
			n_pages = summary["n_pages"].get<int>();
		std::printf("\n=== summary over %d pages ===\n", n_pages);
		if (n_pages == 0)
			return;
		double mask_f1 = 0.0;
		if (summary.contains("mask_prf1")
			&& summary["mask_prf1"].contains("notehead"))
			mask_f1 = summary["mask_prf1"]["notehead"]["f1"].get<double>();
		std::printf("  mask notehead F1: %.4f   (stage 5, splitting)\n",
					mask_f1);
		std::printf("  mAP @IoU=%g: %.4f   (stage 5, detection)\n",
					iou_threshold, summary.value("mAP", 0.0));
		std::printf("  accuracy:         %.4f   "
					"(stage 5, classification)\n",
					summary.value("accuracy", 0.0));
		std::printf("  edge P/R/F1:      %.4f / %.4f / %.4f   "
					"(stage 6, reconstruction)\n",
					summary.value("edge_precision", 0.0),
					summary.value("edge_recall", 0.0),
					summary.value("edge_f1", 0.0));
		std::printf("  SER:              %.4f   (end-to-end, duration)\n",
					summary.value("ser", 0.0));
		std::printf("  pitch SER:        %.4f   (end-to-end, pitch)\n",
					summary.value("pitch_ser", 0.0));
	}

	struct Args
	{
		std::string split = "test";
		int limit = 10;
		double iou_threshold = 0.3;
		std::filesystem::path output = "output/pipeline_eval";
		bool write_json_pages = false;
		std::filesystem::path viz_dir;
		std::optional<unsigned int> seed;
		std::filesystem::path data_dir = "../prototype/data/ds2_dense";
		std::filesystem::path page_ids_file;
	};

	bool parse_args(int argc, char* argv[], Args& args)
	{
		for (int i = 1; i < argc; ++i)
		{
			std::string arg = argv[i];
			auto next = [&](const char* name) -> std::string {
				if (i + 1 >= argc)
				{
					std::fprintf(stderr, "missing value for %s\n", name);
					std::exit(1);
				}
				return argv[++i];
			};
			if (arg == "--split")
				args.split = next("--split");
			else if (arg == "--limit")
				args.limit = std::stoi(next("--limit"));
			else if (arg == "--iou-threshold")
				args.iou_threshold = std::stod(next("--iou-threshold"));
			else if (arg == "--output")
				args.output = next("--output");
			else if (arg == "--data")
				args.data_dir = next("--data");
			else if (arg == "--write-json-pages")
				args.write_json_pages = true;
			else if (arg == "--viz-dir")
				args.viz_dir = next("--viz-dir");
			else if (arg == "--seed")
				args.seed =
					static_cast<unsigned int>(std::stoul(next("--seed")));
			else if (arg == "--page-ids-file")
				args.page_ids_file = next("--page-ids-file");
			else if (arg == "--help" || arg == "-h")
			{
				std::printf(
					"Usage: omr_pipeline_eval [options]\n"
					"  --split <train|test>          (default: test)\n"
					"  --limit <int>                 (default: 10)\n"
					"  --iou-threshold <float>       (default: 0.3)\n"
					"  --output <dir>                (default: "
					"output/pipeline_eval)\n"
					"  --data <dir>                  (default: "
					"../prototype/data/ds2_dense)\n"
					"  --write-json-pages\n"
					"  --viz-dir <dir>                dump per-page viz "
					"artifacts (PNG masks + overlay.json)\n"
					"  --seed <int>\n"
					"  --page-ids-file <tsv>         (band\\timage_id\\t"
					"python_n_crops)\n");
				return false;
			}
			else
			{
				std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
				return false;
			}
		}
		return true;
	}

} // namespace

int main(int argc, char* argv[])
{
	Args args;
	if (!parse_args(argc, argv, args))
		return 1;

	omr::DeepScoresDataset dataset(args.data_dir, args.split);
	omr::PrimitiveSplitter splitter;
	std::filesystem::create_directories(args.output);

	std::vector<json> pages;
	std::vector<std::string> requested_page_ids;

	if (!args.page_ids_file.empty())
	{
		auto manifest = read_page_ids_file(args.page_ids_file);
		std::printf("evaluating %zu pages from manifest\n", manifest.size());
		std::unordered_set<std::string> wanted;
		std::map<std::string, std::string> band_by_id;
		for (const auto& e : manifest)
		{
			requested_page_ids.push_back(e.image_id);
			wanted.insert(e.image_id);
			band_by_id[e.image_id] = e.band;
		}

		std::map<std::string, std::vector<omr::CropSample>> grouped;
		std::map<std::string, double> discover_ms_by_id;
		for (const auto& image_id : requested_page_ids)
		{
			const auto t0 = clock_type::now();
			grouped[image_id] = dataset.discover_page_crops(image_id);
			discover_ms_by_id[image_id] = elapsed_ms(t0, clock_type::now());
		}

		for (const auto& image_id : requested_page_ids)
		{
			const std::string band = band_by_id[image_id];
			auto it = grouped.find(image_id);
			std::vector<omr::CropSample> samples;
			if (it != grouped.end())
				samples = std::move(it->second);
			try
			{
				std::vector<omr::pipeline::CropResult> crop_results_out;
				const auto t1 = clock_type::now();
				auto page = evaluate_page(
					dataset, splitter, image_id, samples, args.iou_threshold,
					args.data_dir, band, nullptr,
					args.viz_dir.empty() ? nullptr : &crop_results_out);
				set_timings(page, discover_ms_by_id[image_id],
							elapsed_ms(t1, clock_type::now()));
				pages.push_back(page);
				print_row(page, 12);
				if (args.write_json_pages)
				{
					auto out_path =
						args.output / ("page_" + image_id + ".json");
					std::ofstream f(out_path);
					if (!f)
						throw std::runtime_error("failed to open "
												 + out_path.string());
					f << page.dump(2);
				}
				if (!args.viz_dir.empty())
				{
					dump_page_artifacts(dataset, image_id, samples,
										crop_results_out, args.viz_dir);
				}
			}
			catch (const std::exception& exc)
			{
				std::fprintf(stderr, "[skip] image %s: %s\n", image_id.c_str(),
							 exc.what());
				json stub;
				stub["image_id"] = image_id;
				if (dataset.images.count(image_id))
					stub["filename"] = dataset.images.at(image_id).filename;
				else
					stub["filename"] = "";
				stub["band"] = band;
				stub["n_crops"] = 0;
				stub["mask_prf1"] = json::object();
				stub["mAP"] = 0.0;
				stub["per_class_ap"] = json::object();
				stub["accuracy"] = 0.0;
				stub["confusion_matrix"] = json::array();
				stub["edge_precision"] = 0.0;
				stub["edge_recall"] = 0.0;
				stub["edge_f1"] = 0.0;
				stub["ser"] = 0.0;
				stub["pitch_ser"] = 0.0;
				stub["n_pred"] = 0;
				stub["n_gt"] = 0;
				stub["pred_duration_sequence"] = json::array();
				stub["gt_duration_sequence"] = json::array();
				stub["pred_pitch_sequence"] = json::array();
				stub["gt_pitch_sequence"] = json::array();
				stub["error"] = exc.what();
				set_timings(stub, discover_ms_by_id[image_id], 0.0);
				pages.push_back(stub);
				print_row(stub, 12);
			}
		}
	}
	else
	{
		std::map<std::string, std::vector<omr::CropSample>> grouped;
		std::map<std::string, double> discover_ms_by_id;
		std::vector<std::string> ids;
		for (const auto& [id, image] : dataset.images)
		{
			(void)image;
			ids.push_back(id);
		}
		std::sort(ids.begin(), ids.end());
		if (args.seed)
			std::shuffle(ids.begin(), ids.end(), std::mt19937(*args.seed));
		if (args.limit > 0 && static_cast<int>(ids.size()) > args.limit)
			ids.resize(args.limit);
		for (const auto& id : ids)
		{
			const auto t0 = clock_type::now();
			grouped[id] = dataset.discover_page_crops(id);
			discover_ms_by_id[id] = elapsed_ms(t0, clock_type::now());
		}

		for (const auto& [image_id, samples] : grouped)
		{
			try
			{
				std::vector<omr::pipeline::CropResult> crop_results_out;
				const auto t1 = clock_type::now();
				auto page = evaluate_page(
					dataset, splitter, image_id, samples, args.iou_threshold,
					args.data_dir, "", nullptr,
					args.viz_dir.empty() ? nullptr : &crop_results_out);
				set_timings(page, discover_ms_by_id[image_id],
							elapsed_ms(t1, clock_type::now()));
				pages.push_back(page);
				print_row(page, 12);
				if (args.write_json_pages)
				{
					auto out_path =
						args.output / ("page_" + image_id + ".json");
					std::ofstream f(out_path);
					if (!f)
						throw std::runtime_error("failed to open "
												 + out_path.string());
					f << page.dump(2);
				}
				if (!args.viz_dir.empty())
				{
					dump_page_artifacts(dataset, image_id, samples,
										crop_results_out, args.viz_dir);
				}
			}
			catch (const std::exception& exc)
			{
				std::fprintf(stderr, "[skip] image %s: %s\n", image_id.c_str(),
							 exc.what());
			}
		}
	}

	json summary = aggregate(pages);
	summary["classes"] = CLASSES;
	summary["iou_threshold"] = args.iou_threshold;
	summary["inference_ground_truth_free"] = true;
	summary["crop_discovery"] = "image_connected_components";
	if (!requested_page_ids.empty())
	{
		summary["requested_page_ids"] = requested_page_ids;
		json eval_ids = json::array();
		for (const auto& p : pages)
			eval_ids.push_back(p["image_id"].get<std::string>());
		summary["evaluated_page_ids"] = eval_ids;
	}
	auto summary_path = args.output / "report.json";
	{
		std::ofstream f(summary_path);
		if (!f)
			throw std::runtime_error("failed to open " + summary_path.string());
		f << summary.dump(2);
	}

	print_summary(summary, args.iou_threshold);
	std::printf("\nreport written to %s\n", summary_path.c_str());
	return 0;
}
