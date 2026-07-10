#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
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

	json evaluate_page(const omr::DeepScoresDataset& dataset,
					   const omr::PrimitiveSplitter& splitter,
					   const std::string& image_id,
					   const std::vector<omr::CropSample>& samples,
					   double iou_threshold,
					   const std::filesystem::path& data_dir)
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
			std::map<std::string, omr::Mask> truth_map(
				sample.truth_masks.begin(), sample.truth_masks.end());
			auto scores = omr::evaluate_split(cr.split, truth_map,
											  &sample.staffless_mask);
			for (auto& [kind, score] : scores)
				mask_scores_per_kind[kind].push_back(score);
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
		omr::PageIO truth = omr::derive_ground_truth(
			dataset, image_id, std::nullopt,
			omr::pipeline::make_crop_bbox_filter(samples), staff_rows);

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
		page["n_crops"] = static_cast<int>(samples.size());
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
		page["pred_duration_sequence"] = predicted.duration_sequence;
		page["gt_duration_sequence"] = truth.duration_sequence;
		page["pred_pitch_sequence"] = predicted.pitch_sequence;
		page["gt_pitch_sequence"] = truth.pitch_sequence;
		return page;
	}

	json aggregate(const std::vector<json>& pages)
	{
		json summary = json::object();
		if (pages.empty())
			return summary;

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
		std::optional<unsigned int> seed;
		std::filesystem::path data_dir = "../prototype/data/ds2_dense";
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
			else if (arg == "--seed")
				args.seed =
					static_cast<unsigned int>(std::stoul(next("--seed")));
			else if (arg == "--help" || arg == "-h")
			{
				std::printf("Usage: omr_pipeline_eval [options]\n"
							"  --split <train|test>          (default: test)\n"
							"  --limit <int>                 (default: 10)\n"
							"  --iou-threshold <float>       (default: 0.3)\n"
							"  --output <dir>                (default: "
							"output/pipeline_eval)\n"
							"  --data <dir>                  (default: "
							"../prototype/data/ds2_dense)\n"
							"  --write-json-pages\n"
							"  --seed <int>\n");
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

	auto crops = dataset.iter_supported_crops(
		std::nullopt, args.seed,
		args.limit > 0 ? std::optional<int>(args.limit) : std::nullopt);

	std::map<std::string, std::vector<omr::CropSample>> grouped;
	for (auto& sample : crops)
	{
		grouped[sample.image_id].push_back(std::move(sample));
		if (static_cast<int>(grouped.size()) >= args.limit)
			break;
	}

	std::vector<json> pages;
	for (const auto& [image_id, samples] : grouped)
	{
		try
		{
			auto page = evaluate_page(dataset, splitter, image_id, samples,
									  args.iou_threshold, args.data_dir);
			pages.push_back(page);
			print_row(page, 12);
			if (args.write_json_pages)
			{
				auto out_path = args.output / ("page_" + image_id + ".json");
				std::ofstream f(out_path);
				if (!f)
					throw std::runtime_error("failed to open "
											 + out_path.string());
				f << page.dump(2);
			}
		}
		catch (const std::exception& exc)
		{
			std::fprintf(stderr, "[skip] image %s: %s\n", image_id.c_str(),
						 exc.what());
		}
	}

	json summary = aggregate(pages);
	summary["classes"] = CLASSES;
	summary["iou_threshold"] = args.iou_threshold;
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
