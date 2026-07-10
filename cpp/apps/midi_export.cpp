#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "deepscores/dataset.h"
#include "deepscores/io.h"
#include "midi/midi.h"
#include "pipeline_common.h"
#include "symbols/splitter.h"

namespace
{
	using json = nlohmann::ordered_json;

	const std::vector<std::string> CLEAN_PAGE_IDS = {
		"1339", "1316", "966", "1257", "84", "1317",
	};

	omr::PageIO predict_page(const omr::DeepScoresDataset& dataset,
							 const omr::PrimitiveSplitter& splitter,
							 const std::string& image_id,
							 const std::vector<omr::CropSample>& samples,
							 const std::filesystem::path& data_dir)
	{
		const omr::DatasetImage& image = samples.empty()
			? dataset.images.at(image_id)
			: dataset.images.at(samples.front().image_id);
		const auto image_path = data_dir / "images" / image.filename;
		std::vector<double> staff_rows;
		if (std::filesystem::exists(image_path))
			staff_rows = omr::pipeline::recover_staff_rows(image_path);
		const double staff_line = omr::pipeline::compute_staff_line(samples);

		std::vector<omr::pipeline::CropResult> crop_results;
		std::vector<std::vector<double>> staff_lines_per_crop;
		for (const auto& sample : samples)
		{
			auto cr = omr::pipeline::process_crop(splitter, sample, staff_rows,
												  staff_line);
			staff_lines_per_crop.push_back(cr.system_lines);
			crop_results.push_back(std::move(cr));
		}

		std::vector<omr::PredictedCropResult> crop_refs;
		crop_refs.reserve(crop_results.size());
		for (std::size_t i = 0; i < crop_results.size(); ++i)
		{
			crop_refs.push_back({ &crop_results[i].split,
								  &crop_results[i].graph, samples[i].origin });
		}

		return omr::build_predicted_page(image, crop_refs, staff_lines_per_crop,
										 staff_line);
	}

	struct Args
	{
		std::string split = "test";
		int limit = 6;
		std::filesystem::path output = "output/midi";
		int tempo = 120;
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
			else if (arg == "--output")
				args.output = next("--output");
			else if (arg == "--data")
				args.data_dir = next("--data");
			else if (arg == "--tempo")
				args.tempo = std::stoi(next("--tempo"));
			else if (arg == "--help" || arg == "-h")
			{
				std::printf("Usage: omr_midi_export [options]\n"
							"  --split <train|test>   (default: test)\n"
							"  --limit <int>          (default: 6)\n"
							"  --output <dir>         (default: output/midi)\n"
							"  --data <dir>           (default: "
							"../prototype/data/ds2_dense)\n"
							"  --tempo <int>          (default: 120)\n");
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

	const int limit =
		std::min(args.limit, static_cast<int>(CLEAN_PAGE_IDS.size()));
	std::unordered_set<std::string> wanted(CLEAN_PAGE_IDS.begin(),
										   CLEAN_PAGE_IDS.begin() + limit);

	auto crops = dataset.iter_supported_crops(std::nullopt, std::nullopt,
											  std::nullopt, wanted);
	std::unordered_map<std::string, std::vector<omr::CropSample>> grouped;
	for (auto& sample : crops)
	{
		if (wanted.count(sample.image_id))
			grouped[sample.image_id].push_back(std::move(sample));
	}

	for (int idx = 0; idx < limit; ++idx)
	{
		const std::string& image_id = CLEAN_PAGE_IDS[idx];
		auto it = grouped.find(image_id);
		if (it == grouped.end() || it->second.empty())
		{
			std::printf("[skip] image %s: no supported crops\n",
						image_id.c_str());
			continue;
		}
		const auto& samples = it->second;
		try
		{
			omr::PageIO page = predict_page(dataset, splitter, image_id,
											samples, args.data_dir);

			auto mid_path = args.output / (image_id + ".mid");
			int written =
				omr::write_midi(page.duration_sequence, page.pitch_sequence,
								mid_path, 480, args.tempo);

			json sidecar;
			sidecar["image_id"] = image_id;
			sidecar["filename"] = page.filename;
			sidecar["n_crops"] = static_cast<int>(samples.size());
			sidecar["duration_sequence"] = page.duration_sequence;
			sidecar["pitch_sequence"] = page.pitch_sequence;
			sidecar["midi_file"] = mid_path.string();
			sidecar["notes_written"] = written;
			sidecar["tempo_bpm"] = args.tempo;

			auto json_path = args.output / (image_id + ".json");
			std::ofstream f(json_path);
			if (!f)
				throw std::runtime_error("failed to open "
										 + json_path.string());
			f << sidecar.dump(2);

			std::printf("image %s: %d notes, %zu tokens -> %s.mid\n",
						image_id.c_str(), written,
						page.duration_sequence.size(), image_id.c_str());
		}
		catch (const std::exception& exc)
		{
			std::fprintf(stderr, "[skip] image %s: %s\n", image_id.c_str(),
						 exc.what());
		}
	}
	return 0;
}
