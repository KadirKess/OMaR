#include "deepscores/dataset.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "preprocessing/preprocessor.h"

namespace omr
{
	namespace
	{
		const std::unordered_map<std::string, std::string>
			TARGET_CATEGORY_NAMES{
				{ "noteheadBlackOnLine", "notehead" },
				{ "noteheadBlackInSpace", "notehead" },
				{ "stem", "stem" },
				{ "beam", "beam" },
			};

		std::string json_id(const nlohmann::json& value)
		{
			if (value.is_string())
				return value.get<std::string>();
			if (value.is_number_integer())
				return std::to_string(value.get<long long>());
			if (value.is_number_unsigned())
				return std::to_string(value.get<unsigned long long>());
			return value.dump();
		}

		int json_int(const nlohmann::json& object, const char* key,
					 int fallback = 0)
		{
			if (!object.contains(key))
				return fallback;
			const auto& value = object.at(key);
			if (value.is_number_integer())
				return value.get<int>();
			if (value.is_string())
				return std::stoi(value.get<std::string>());
			return fallback;
		}

		BBoxD parse_bbox(const nlohmann::json& value)
		{
			if (!value.is_array() || value.size() < 4)
				throw std::runtime_error(
					"DeepScores annotation missing a_bbox");
			return BBoxD{ value[0].get<double>(), value[1].get<double>(),
						  value[2].get<double>(), value[3].get<double>() };
		}

		double median(std::vector<double> values)
		{
			if (values.empty())
				return 0.0;
			std::sort(values.begin(), values.end());
			const std::size_t mid = values.size() / 2;
			if (values.size() % 2 == 1)
				return values[mid];
			return 0.5 * (values[mid - 1] + values[mid]);
		}

		Mask crop_mask(const Mask& mask, int x0, int y0, int x1, int y1)
		{
			const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
			return Mask(mask.mat()(roi).clone());
		}

		std::vector<std::string> sorted_ids(std::unordered_set<std::string> ids)
		{
			std::vector<std::string> out(ids.begin(), ids.end());
			std::sort(out.begin(), out.end());
			return out;
		}

		bool mat_equals_color(const cv::Mat& semantic, int y, int x, int color)
		{
			switch (semantic.type())
			{
			case CV_8U:
				return semantic.at<std::uint8_t>(y, x) == color;
			case CV_16U:
				return semantic.at<std::uint16_t>(y, x) == color;
			case CV_32S:
				return semantic.at<int>(y, x) == color;
			default:
				return false;
			}
		}

		std::optional<std::vector<cv::Vec3b>>
		read_png_palette(const std::filesystem::path& path)
		{
			std::ifstream file(path, std::ios::binary);
			if (!file)
				return std::nullopt;
			std::vector<std::uint8_t> data(
				(std::istreambuf_iterator<char>(file)),
				std::istreambuf_iterator<char>());
			if (data.size() < 8)
				return std::nullopt;
			const std::uint8_t sig[] = { 0x89, 0x50, 0x4E, 0x47,
										 0x0D, 0x0A, 0x1A, 0x0A };
			if (std::memcmp(data.data(), sig, 8) != 0)
				return std::nullopt;
			std::size_t pos = 8;
			while (pos + 8 <= data.size())
			{
				const std::uint32_t length =
					(static_cast<std::uint32_t>(data[pos]) << 24)
					| (static_cast<std::uint32_t>(data[pos + 1]) << 16)
					| (static_cast<std::uint32_t>(data[pos + 2]) << 8)
					| static_cast<std::uint32_t>(data[pos + 3]);
				pos += 4;
				const char type[5] = { static_cast<char>(data[pos]),
									   static_cast<char>(data[pos + 1]),
									   static_cast<char>(data[pos + 2]),
									   static_cast<char>(data[pos + 3]), 0 };
				pos += 4;
				if (pos + length + 4 > data.size())
					break;
				if (std::strcmp(type, "PLTE") == 0)
				{
					const std::size_t n = length / 3;
					std::vector<cv::Vec3b> palette(n);
					for (std::size_t i = 0; i < n; ++i)
						palette[i] =
							cv::Vec3b(data[pos + i * 3 + 2],
									  data[pos + i * 3 + 1], data[pos + i * 3]);
					return palette;
				}
				pos += length + 4;
			}
			return std::nullopt;
		}

		cv::Mat normalize_semantic(cv::Mat mat,
								   const std::filesystem::path& path)
		{
			if (mat.empty())
				throw std::runtime_error("failed to read semantic mask");
			if (mat.channels() == 1)
				return mat;
			auto palette = read_png_palette(path);
			if (!palette || palette->empty())
			{
				std::vector<cv::Mat> channels;
				cv::split(mat, channels);
				return channels[0];
			}
			std::unordered_map<std::uint32_t, std::uint8_t> lookup;
			lookup.reserve(palette->size());
			for (std::size_t i = 0; i < palette->size(); ++i)
			{
				const auto& bgr = (*palette)[i];
				const std::uint32_t key =
					(static_cast<std::uint32_t>(bgr[0]) << 16)
					| (static_cast<std::uint32_t>(bgr[1]) << 8)
					| static_cast<std::uint32_t>(bgr[2]);
				lookup.emplace(key, static_cast<std::uint8_t>(i));
			}
			cv::Mat1b out = cv::Mat1b::zeros(mat.size());
			for (int y = 0; y < mat.rows; ++y)
			{
				const auto* src = mat.ptr<cv::Vec3b>(y);
				auto* dst = out.ptr<std::uint8_t>(y);
				for (int x = 0; x < mat.cols; ++x)
				{
					const auto& bgr = src[x];
					const std::uint32_t key =
						(static_cast<std::uint32_t>(bgr[0]) << 16)
						| (static_cast<std::uint32_t>(bgr[1]) << 8)
						| static_cast<std::uint32_t>(bgr[2]);
					auto it = lookup.find(key);
					dst[x] = it != lookup.end() ? it->second : 0;
				}
			}
			return out;
		}
	} // namespace

	double bbox_distance(const BBoxD& a, const BBoxD& b)
	{
		const double dx = std::max({ a.x0 - b.x1, b.x0 - a.x1, 0.0 });
		const double dy = std::max({ a.y0 - b.y1, b.y0 - a.y1, 0.0 });
		return std::hypot(dx, dy);
	}

	bool CropDiagnostics::balances() const noexcept
	{
		return rejected_no_supported_notehead + rejected_stacked_head
			+ rejected_semantic_union_empty + rejected_no_ink_touched
			+ yielded_crops
			== components_considered;
	}

	CropDiagnostics&
	CropDiagnostics::operator+=(const CropDiagnostics& other) noexcept
	{
		candidate_target_annotations += other.candidate_target_annotations;
		components_considered += other.components_considered;
		rejected_no_supported_notehead += other.rejected_no_supported_notehead;
		rejected_stacked_head += other.rejected_stacked_head;
		rejected_semantic_union_empty += other.rejected_semantic_union_empty;
		rejected_no_ink_touched += other.rejected_no_ink_touched;
		yielded_crops += other.yielded_crops;
		images_without_noteheads += other.images_without_noteheads;
		return *this;
	}

	DeepScoresDataset::DeepScoresDataset(std::filesystem::path root,
										 const std::string& split)
		: m_root(std::move(root))
	{
		const auto annotation_path = m_root / ("deepscores_" + split + ".json");
		std::ifstream input(annotation_path);
		if (!input)
			throw std::runtime_error("failed to open "
									 + annotation_path.string());
		nlohmann::json data;
		input >> data;

		for (const auto& [id, category] : data.at("categories").items())
		{
			DatasetCategory parsed;
			parsed.id = id;
			parsed.name = category.value("name", "");
			parsed.annotation_set = category.value("annotation_set", "");
			parsed.color = json_int(category, "color");
			parsed.raw = category;
			categories.emplace(parsed.id, parsed);
			if (parsed.annotation_set == "deepscores")
				category_ids[parsed.name] = parsed.id;
		}

		for (const auto& image_json : data.at("images"))
		{
			DatasetImage image;
			image.id = json_id(image_json.at("id"));
			image.filename = image_json.value("filename", "");
			image.width = json_int(image_json, "width");
			image.height = json_int(image_json, "height");
			image.raw = image_json;
			images.emplace(image.id, image);
		}

		for (const auto& [id, ann_json] : data.at("annotations").items())
		{
			DatasetAnnotation ann;
			ann.id = id;
			ann.img_id = json_id(ann_json.at("img_id"));
			for (const auto& cat : ann_json.at("cat_id"))
				ann.cat_id.push_back(json_id(cat));
			ann.a_bbox = parse_bbox(ann_json.at("a_bbox"));
			ann.area = ann_json.value("area", 0);
			ann.raw = ann_json;
			by_image[ann.img_id].push_back(ann.id);
			annotations.emplace(ann.id, ann);
		}
	}

	cv::Mat DeepScoresDataset::semantic_mask(const DatasetImage& image) const
	{
		std::string filename = image.filename;
		const auto pos = filename.rfind(".png");
		if (pos != std::string::npos)
			filename.replace(pos, 4, "_seg.png");
		const auto path = m_root / "segmentation" / filename;
		return normalize_semantic(
			cv::imread(path.string(), cv::IMREAD_UNCHANGED), path);
	}

	Mask DeepScoresDataset::ink_mask(const DatasetImage& image) const
	{
		const auto path = m_root / "images" / image.filename;
		cv::Mat raw = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
		if (raw.empty())
			throw std::runtime_error("failed to read image " + path.string());
		cv::Mat gray;
		if (raw.channels() == 1)
			gray = raw;
		else
			cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
		if (gray.type() != CV_8U)
			gray.convertTo(gray, CV_8U);
		cv::Mat1b out;
		cv::threshold(gray, out, 0, 255,
					  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
		return Mask(std::move(out));
	}

	std::tuple<Mask, Mask, float, float>
	DeepScoresDataset::staffless_ink(const DatasetImage& image) const
	{
		Mask ink = ink_mask(image);
		cv::Mat1b pre_image;
		cv::bitwise_not(ink.mat(), pre_image);
		Preprocessor preprocessor(pre_image);
		const StaffDimensions dims = preprocessor.compute_staff_space_height();
		const auto rows = preprocessor.staffline_detection_projection();
		preprocessor.remove_stafflines(rows);
		Mask staffless(preprocessor.image() == 0);
		const float staff_space = static_cast<float>(dims.staff_space_height
													 + dims.staff_line_height);
		const float staff_line =
			static_cast<float>(std::max(1, dims.staff_line_height));
		return { ink, staffless, staff_space, staff_line };
	}

	Mask
	DeepScoresDataset::category_mask(const cv::Mat& semantic,
									 const DatasetAnnotation& annotation) const
	{
		const auto category_id = annotation.cat_id.at(0);
		const auto cat_it = categories.find(category_id);
		if (cat_it == categories.end())
			throw std::runtime_error("unknown category id " + category_id);
		const int color = cat_it->second.color;
		cv::Mat1b out = cv::Mat1b::zeros(semantic.size());
		const int x0 = std::max(0, static_cast<int>(annotation.a_bbox.x0));
		const int y0 = std::max(0, static_cast<int>(annotation.a_bbox.y0));
		const int x1 =
			std::min(semantic.cols - 1, static_cast<int>(annotation.a_bbox.x1));
		const int y1 =
			std::min(semantic.rows - 1, static_cast<int>(annotation.a_bbox.y1));
		for (int y = y0; y <= y1; ++y)
		{
			std::uint8_t* row = out.ptr<std::uint8_t>(y);
			for (int x = x0; x <= x1; ++x)
				if (mat_equals_color(semantic, y, x, color))
					row[x] = 255;
		}
		return Mask(std::move(out));
	}

	std::tuple<Mask, Mask, DatasetImage>
	DeepScoresDataset::load_page(const std::string& image_id) const
	{
		const auto image = images.at(image_id);
		auto [ink, staffless, staff_space, staff_line] = staffless_ink(image);
		(void)staff_space;
		(void)staff_line;
		return { std::move(ink), std::move(staffless), image };
	}

	std::vector<CropSample>
	DeepScoresDataset::discover_page_crops(const std::string& image_id,
										   double link_scale,
										   double margin_scale) const
	{
		const DatasetImage& image = images.at(image_id);
		auto [ink, staffless, staff_space, staff_line] = staffless_ink(image);
		cv::Mat1b pre_image;
		cv::bitwise_not(ink.mat(), pre_image);
		const Preprocessor staff_detector(pre_image);
		const std::vector<int> staff_rows =
			staff_detector.staffline_detection_projection();
		cv::Mat labels, stats, centroids;
		const int count = cv::connectedComponentsWithStats(
			staffless.mat(), labels, stats, centroids, 8, CV_32S);
		const int minimum_area = std::max(
			2, static_cast<int>(std::lround(0.08 * staff_space * staff_space)));
		const double link_distance = std::max(2.0, link_scale * staff_space);

		struct Region
		{
			int label;
			BBox box;
		};
		std::vector<Region> regions;
		const int staff_margin =
			static_cast<int>(std::lround(2.0 * staff_space));
		for (int label = 1; label < count; ++label)
		{
			if (stats.at<int>(label, cv::CC_STAT_AREA) < minimum_area)
				continue;
			const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
			const int y = stats.at<int>(label, cv::CC_STAT_TOP);
			const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
			const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
			const bool near_staff = staff_rows.empty()
				|| std::any_of(staff_rows.begin(), staff_rows.end(),
							   [&](int row) {
								   return row >= y - staff_margin
									   && row < y + h + staff_margin;
							   });
			if (!near_staff)
				continue;
			regions.push_back({ label, { x, y, x + w, y + h } });
		}

		auto distance = [](const BBox& a, const BBox& b) {
			const int dx = std::max({ 0, a.x0 - b.x1, b.x0 - a.x1 });
			const int dy = std::max({ 0, a.y0 - b.y1, b.y0 - a.y1 });
			return std::hypot(static_cast<double>(dx), static_cast<double>(dy));
		};
		std::vector<int> parent(regions.size());
		for (std::size_t i = 0; i < parent.size(); ++i)
			parent[i] = static_cast<int>(i);
		auto find = [&](int value) {
			int root = value;
			while (parent[root] != root)
				root = parent[root];
			while (parent[value] != value)
			{
				const int next = parent[value];
				parent[value] = root;
				value = next;
			}
			return root;
		};
		for (std::size_t i = 0; i < regions.size(); ++i)
			for (std::size_t j = i + 1; j < regions.size(); ++j)
				if (distance(regions[i].box, regions[j].box) <= link_distance)
				{
					const int ri = find(static_cast<int>(i));
					const int rj = find(static_cast<int>(j));
					if (ri != rj)
						parent[rj] = ri;
				}

		std::unordered_map<int, std::vector<int>> groups;
		for (std::size_t i = 0; i < regions.size(); ++i)
			groups[find(static_cast<int>(i))].push_back(static_cast<int>(i));
		const int margin =
			static_cast<int>(std::lround(margin_scale * staff_space));
		std::vector<CropSample> samples;
		for (const auto& [root, members] : groups)
		{
			(void)root;
			BBox box = regions[members.front()].box;
			std::unordered_set<int> group_labels;
			for (int index : members)
			{
				const auto& region = regions[index];
				group_labels.insert(region.label);
				box.x0 = std::min(box.x0, region.box.x0);
				box.y0 = std::min(box.y0, region.box.y0);
				box.x1 = std::max(box.x1, region.box.x1);
				box.y1 = std::max(box.y1, region.box.y1);
			}
			const int x0 = std::max(0, box.x0 - margin);
			const int y0 = std::max(0, box.y0 - margin);
			const int x1 = std::min(ink.cols(), box.x1 + margin);
			const int y1 = std::min(ink.rows(), box.y1 + margin);
			CropSample sample;
			sample.original_mask = crop_mask(ink, x0, y0, x1, y1);
			sample.staffless_mask =
				crop_mask(is_in(labels, group_labels), x0, y0, x1, y1);
			sample.staff_space = staff_space;
			sample.staff_line = staff_line;
			sample.image_id = image_id;
			sample.origin = { x0, y0 };
			samples.push_back(std::move(sample));
		}
		std::sort(
			samples.begin(), samples.end(),
			[](const auto& a, const auto& b) { return a.origin < b.origin; });
		return samples;
	}

	std::vector<CropSample> DeepScoresDataset::iter_supported_crops(
		std::optional<int> limit, std::optional<unsigned int> shuffle_seed,
		std::optional<int> max_pages,
		std::optional<std::unordered_set<std::string>> image_id_filter,
		CropDiagnosticsByImage* diagnostics) const
	{
		std::vector<CropSample> samples;
		if (limit && *limit <= 0)
			return samples;
		if (max_pages && *max_pages <= 0)
			return samples;

		std::unordered_set<std::string> target_ids;
		for (const auto& [name, output] : TARGET_CATEGORY_NAMES)
		{
			(void)output;
			if (auto it = category_ids.find(name); it != category_ids.end())
				target_ids.insert(it->second);
		}
		const auto staff_it = category_ids.find("staff");

		std::vector<std::string> image_ids;
		image_ids.reserve(images.size());
		for (const auto& [id, image] : images)
		{
			(void)image;
			image_ids.push_back(id);
		}
		std::sort(image_ids.begin(), image_ids.end(),
				  [](const std::string& a, const std::string& b) {
					  try
					  {
						  return std::stoll(a) < std::stoll(b);
					  }
					  catch (...)
					  {
						  return a < b;
					  }
				  });
		if (shuffle_seed)
		{
			std::mt19937 rng(*shuffle_seed);
			std::shuffle(image_ids.begin(), image_ids.end(), rng);
		}

		std::unordered_set<std::string> pages_seen;

		for (const auto& image_id : image_ids)
		{
			if (image_id_filter && !image_id_filter->count(image_id))
				continue;
			CropDiagnostics* page_diagnostics =
				diagnostics ? &(*diagnostics)[image_id] : nullptr;
			const DatasetImage& image = images.at(image_id);
			const auto by_it = by_image.find(image_id);
			if (by_it == by_image.end())
				continue;

			std::vector<std::string> targets;
			std::vector<std::string> heads;
			for (const auto& aid : by_it->second)
			{
				const auto& ann = annotations.at(aid);
				if (ann.cat_id.empty() || !target_ids.count(ann.cat_id[0]))
					continue;
				targets.push_back(aid);
				const auto& cat = categories.at(ann.cat_id[0]);
				if (TARGET_CATEGORY_NAMES.at(cat.name) == "notehead")
					heads.push_back(aid);
			}
			if (page_diagnostics)
				page_diagnostics->candidate_target_annotations =
					static_cast<int>(targets.size());
			if (heads.empty())
			{
				if (page_diagnostics)
					page_diagnostics->images_without_noteheads = 1;
				continue;
			}

			std::vector<double> head_heights;
			head_heights.reserve(heads.size());
			for (const auto& aid : heads)
			{
				const auto& bbox = annotations.at(aid).a_bbox;
				head_heights.push_back(bbox.y1 - bbox.y0);
			}
			const double staff_space = std::max(4.0, median(head_heights));
			const double link_distance = std::max(2.0, 0.35 * staff_space);

			std::unordered_map<std::string, std::vector<std::string>> graph;
			for (std::size_t i = 0; i < targets.size(); ++i)
			{
				graph[targets[i]];
				for (std::size_t j = i + 1; j < targets.size(); ++j)
				{
					if (bbox_distance(annotations.at(targets[i]).a_bbox,
									  annotations.at(targets[j]).a_bbox)
						<= link_distance)
					{
						graph[targets[i]].push_back(targets[j]);
						graph[targets[j]].push_back(targets[i]);
					}
				}
			}

			std::unordered_set<std::string> seen;
			std::optional<cv::Mat> semantic;
			std::optional<Mask> ink;
			std::optional<cv::Mat> labels;

			for (const auto& seed : targets)
			{
				if (seen.count(seed))
					continue;
				std::vector<std::string> stack{ seed };
				std::unordered_set<std::string> component_ids;
				while (!stack.empty())
				{
					std::string current = stack.back();
					stack.pop_back();
					if (!component_ids.insert(current).second)
						continue;
					for (const auto& next : graph[current])
						stack.push_back(next);
				}
				seen.insert(component_ids.begin(), component_ids.end());

				std::vector<std::string> component = sorted_ids(component_ids);
				std::vector<std::string> kinds;
				kinds.reserve(component.size());
				int stem_count = 0;
				int beam_count = 0;
				for (const auto& aid : component)
				{
					const auto& ann = annotations.at(aid);
					const auto& cat = categories.at(ann.cat_id[0]);
					const std::string kind = TARGET_CATEGORY_NAMES.at(cat.name);
					kinds.push_back(kind);
					if (kind == "stem")
						++stem_count;
					else if (kind == "beam")
						++beam_count;
				}
				(void)stem_count;
				(void)beam_count;
				if (page_diagnostics)
					++page_diagnostics->components_considered;

				std::vector<std::pair<double, double>> centers;
				for (std::size_t i = 0; i < component.size(); ++i)
				{
					if (kinds[i] != "notehead")
						continue;
					const auto& bbox = annotations.at(component[i]).a_bbox;
					centers.emplace_back((bbox.x0 + bbox.x1) / 2.0,
										 bbox.x1 - bbox.x0);
				}
				bool stacked_heads = false;
				for (std::size_t i = 0; i < centers.size(); ++i)
				{
					for (std::size_t j = i + 1; j < centers.size(); ++j)
					{
						if (std::abs(centers[i].first - centers[j].first) < 0.6
								* std::min(centers[i].second,
										   centers[j].second))
							stacked_heads = true;
					}
				}
				if (stacked_heads)
				{
					if (page_diagnostics)
						++page_diagnostics->rejected_stacked_head;
					continue;
				}

				if (!semantic)
				{
					semantic = semantic_mask(image);
					ink = ink_mask(image);
					Mask staff_mask(ink->rows(), ink->cols());
					if (staff_it != category_ids.end())
					{
						const int color = categories.at(staff_it->second).color;
						cv::Mat1b staff = cv::Mat1b::zeros(semantic->size());
						for (int y = 0; y < semantic->rows; ++y)
						{
							std::uint8_t* row = staff.ptr<std::uint8_t>(y);
							for (int x = 0; x < semantic->cols; ++x)
								if (mat_equals_color(*semantic, y, x, color))
									row[x] = 255;
						}
						staff_mask = Mask(std::move(staff));
					}
					cv::Mat label_mat;
					cv::connectedComponents(((*ink) & ~staff_mask).mat(),
											label_mat, 8, CV_32S);
					labels = label_mat;
				}

				std::unordered_map<std::string, Mask> truth_full;
				for (const auto& kind : { "notehead", "stem", "beam" })
					truth_full.emplace(kind, Mask(ink->rows(), ink->cols()));

				for (std::size_t i = 0; i < component.size(); ++i)
				{
					const auto& ann = annotations.at(component[i]);
					const std::string& kind = kinds[i];
					const Mask category = category_mask(*semantic, ann);
					const int x0 = std::max(0, static_cast<int>(ann.a_bbox.x0));
					const int y0 = std::max(0, static_cast<int>(ann.a_bbox.y0));
					const int x1 = std::min(ink->cols() - 1,
											static_cast<int>(ann.a_bbox.x1));
					const int y1 = std::min(ink->rows() - 1,
											static_cast<int>(ann.a_bbox.y1));
					for (int y = y0; y <= y1; ++y)
					{
						for (int x = x0; x <= x1; ++x)
						{
							if (category.at(y, x))
								truth_full.at(kind).set(y, x, 255);
						}
					}
				}

				Mask union_mask(ink->rows(), ink->cols());
				for (const auto& [kind, mask] : truth_full)
				{
					(void)kind;
					union_mask = union_mask | mask;
				}
				if (!union_mask.any())
				{
					if (page_diagnostics)
						++page_diagnostics->rejected_semantic_union_empty;
					continue;
				}
				std::unordered_set<int> touched;
				for (int y = 0; y < union_mask.rows(); ++y)
				{
					const int* label_row = labels->ptr<int>(y);
					for (int x = 0; x < union_mask.cols(); ++x)
					{
						if (union_mask.at(y, x) && label_row[x] != 0)
							touched.insert(label_row[x]);
					}
				}
				if (touched.empty())
				{
					if (page_diagnostics)
						++page_diagnostics->rejected_no_ink_touched;
					continue;
				}
				const Mask compound = is_in(*labels, touched);

				auto bbox = union_mask.bbox();
				if (!bbox)
					continue;
				const int margin =
					static_cast<int>(std::lround(1.5 * staff_space));
				const int x0 = std::max(0, bbox->x0 - margin);
				const int x1 = std::min(ink->cols(), bbox->x1 + margin);
				const int y0 = std::max(0, bbox->y0 - margin);
				const int y1 = std::min(ink->rows(), bbox->y1 + margin);

				std::unordered_map<std::string, Mask> crop_truth;
				for (const auto& kind : { "notehead", "stem", "beam" })
					crop_truth.emplace(
						kind, crop_mask(truth_full.at(kind), x0, y0, x1, y1));
				CropSample sample;
				sample.original_mask = crop_mask(*ink, x0, y0, x1, y1);
				sample.staffless_mask = crop_mask(compound, x0, y0, x1, y1);
				sample.truth_masks = std::move(crop_truth);
				sample.staff_space = static_cast<float>(staff_space);
				sample.staff_line = static_cast<float>(
					std::max(1.0, std::round(0.15 * staff_space)));
				sample.image_id = image_id;
				sample.annotation_ids = sorted_ids(component_ids);
				sample.origin = { x0, y0 };
				samples.push_back(std::move(sample));
				if (page_diagnostics)
					++page_diagnostics->yielded_crops;
				if (limit && static_cast<int>(samples.size()) >= *limit)
					return samples;
				bool new_page = pages_seen.insert(image_id).second;
				if (max_pages && new_page
					&& static_cast<int>(pages_seen.size()) >= *max_pages)
					return samples;
			}
		}

		return samples;
	}

} // namespace omr
