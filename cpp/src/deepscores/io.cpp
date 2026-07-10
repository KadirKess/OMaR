#include "deepscores/io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "reconstruction/semantics.h"

namespace omr
{
	namespace
	{
		const std::unordered_map<PrimitiveKind, std::string> PRIMITIVE_TO_CAT{
			{ PrimitiveKind::STEM, "stem" },
			{ PrimitiveKind::BEAM, "beam" },
		};

		std::optional<double>
		nearest_system_bottom(std::vector<double> staff_rows, double y_center)
		{
			std::sort(staff_rows.begin(), staff_rows.end());
			if (staff_rows.empty())
				return std::nullopt;
			if (staff_rows.size() <= 5)
				return staff_rows.back();
			double best_bottom = 0.0;
			double best_dist = std::numeric_limits<double>::infinity();
			for (std::size_t i = 0; i + 4 < staff_rows.size(); ++i)
			{
				const double center = (staff_rows[i] + staff_rows[i + 4]) / 2.0;
				const double dist = std::abs(center - y_center);
				if (dist < best_dist)
				{
					best_dist = dist;
					best_bottom = staff_rows[i + 4];
				}
			}
			return best_bottom;
		}

		std::string step_to_pitch(int step, char base_letter, int base_octave)
		{
			const std::string letters = "CDEFGAB";
			const int base = static_cast<int>(letters.find(base_letter));
			const int total = base + step;
			const int letter_index = ((total % 7) + 7) % 7;
			const int octave = base_octave
				+ static_cast<int>(std::floor(static_cast<double>(total)
											  / 7.0));
			return std::string(1, letters[letter_index])
				+ std::to_string(octave);
		}

		std::string duration_token(double duration)
		{
			if (std::abs(duration - QUARTER) < 1e-12)
				return "quarter";
			if (std::abs(duration - EIGHTH) < 1e-12)
				return "eighth";
			return "?";
		}

		const Primitive* first_head(const NodePtr& node)
		{
			if (auto head = node->get<const Primitive*>("notehead"))
				return *head;
			if (auto heads =
					node->get<std::vector<const Primitive*>>("noteheads"))
			{
				if (!heads->empty())
					return (*heads)[0];
			}
			return nullptr;
		}

		double node_x(const NodePtr& node, int origin_x)
		{
			const Primitive* head = first_head(node);
			if (head == nullptr)
				return static_cast<double>(origin_x);
			return static_cast<double>(head->bbox.x0 + origin_x);
		}

		std::vector<std::pair<double, std::string>>
		duration_sequence_from_graph(const NotationGraph& graph,
									 std::pair<int, int> origin)
		{
			std::vector<NodePtr> nodes;
			for (const auto& [id, node] : graph.nodes)
			{
				(void)id;
				if (node->kind == NodeKind::NOTE
					|| node->kind == NodeKind::CHORD
					|| node->kind == NodeKind::BEAMED_GROUP)
					nodes.push_back(node);
			}
			std::sort(nodes.begin(), nodes.end(),
					  [](const NodePtr& a, const NodePtr& b) {
						  return a->get_or<int>("x_order", 0)
							  < b->get_or<int>("x_order", 0);
					  });

			std::vector<std::pair<double, std::string>> out;
			for (const auto& node : nodes)
			{
				if (node->kind == NodeKind::BEAMED_GROUP)
				{
					if (auto members =
							node->get<std::vector<NodePtr>>("members"))
					{
						for (const auto& member : *members)
							out.emplace_back(
								node_x(member, origin.first),
								duration_token(
									member->get_or<double>("duration", -1.0)));
					}
				}
				else
				{
					out.emplace_back(
						node_x(node, origin.first),
						duration_token(node->get_or<double>("duration", -1.0)));
				}
			}
			return out;
		}

		std::vector<std::pair<double, std::string>>
		pitch_sequence_from_graph(const NotationGraph& graph,
								  std::pair<int, int> origin)
		{
			std::vector<NodePtr> nodes;
			for (const auto& [id, node] : graph.nodes)
			{
				(void)id;
				if (node->kind == NodeKind::NOTE
					|| node->kind == NodeKind::CHORD
					|| node->kind == NodeKind::BEAMED_GROUP)
					nodes.push_back(node);
			}
			std::sort(nodes.begin(), nodes.end(),
					  [](const NodePtr& a, const NodePtr& b) {
						  return a->get_or<int>("x_order", 0)
							  < b->get_or<int>("x_order", 0);
					  });

			std::vector<std::pair<double, std::string>> out;
			for (const auto& node : nodes)
			{
				if (node->kind == NodeKind::BEAMED_GROUP)
				{
					if (auto members =
							node->get<std::vector<NodePtr>>("members"))
					{
						for (const auto& member : *members)
							out.emplace_back(
								node_x(member, origin.first),
								member->get_or<std::string>("pitch", "?"));
					}
				}
				else
				{
					out.emplace_back(node_x(node, origin.first),
									 node->get_or<std::string>("pitch", "?"));
				}
			}
			return out;
		}

		std::pair<std::vector<Annotation>, std::unordered_map<int, std::string>>
		annotations_from_split(const PrimitiveSplit& split, int crop_index,
							   std::pair<int, int> origin,
							   const std::vector<double>& staff_lines,
							   double staff_line)
		{
			std::vector<Annotation> out;
			std::unordered_map<int, std::string> id_map;
			for (int i = 0; i < static_cast<int>(split.primitives.size()); ++i)
			{
				const Primitive& prim = split.primitives[i];
				if (prim.kind == PrimitiveKind::UNKNOWN)
					continue;
				BBoxD page_bbox{
					static_cast<double>(prim.bbox.x0 + origin.first),
					static_cast<double>(prim.bbox.y0 + origin.second),
					static_cast<double>(prim.bbox.x1 + origin.first),
					static_cast<double>(prim.bbox.y1 + origin.second),
				};
				std::string cat_name;
				if (prim.kind == PrimitiveKind::NOTEHEAD)
					cat_name =
						notehead_cat_name(page_bbox, staff_lines, staff_line);
				else
					cat_name = PRIMITIVE_TO_CAT.at(prim.kind);

				const std::string ann_id = "pred-" + std::to_string(crop_index)
					+ "-" + std::to_string(i);
				out.push_back(Annotation{ ann_id, page_bbox, cat_name,
										  prim.mask.count_nonzero(),
										  prim.confidence });
				id_map[i] = ann_id;
			}
			return { out, id_map };
		}

		std::vector<GraphEdgeSpec>
		edges_from_split(const PrimitiveSplit& split,
						 const std::unordered_map<int, std::string>& id_map)
		{
			std::vector<GraphEdgeSpec> out;
			for (const auto& [head_idx, stem_idx] : split.head_to_stem)
			{
				if (id_map.count(head_idx) && id_map.count(stem_idx))
					out.push_back({ "head_to_stem", id_map.at(head_idx),
									id_map.at(stem_idx) });
			}
			for (const auto& [stem_idx, beam_idx] : split.stem_to_beam)
			{
				if (id_map.count(stem_idx) && id_map.count(beam_idx))
					out.push_back({ "stem_to_beam", id_map.at(stem_idx),
									id_map.at(beam_idx) });
			}
			return out;
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

		nlohmann::json bbox_json(const BBoxD& bbox)
		{
			return nlohmann::json::array(
				{ bbox.x0, bbox.y0, bbox.x1, bbox.y1 });
		}

		nlohmann::json obox_json(const BBoxD& bbox)
		{
			return nlohmann::json::array({ bbox.x0, bbox.y0, bbox.x1, bbox.y0,
										   bbox.x1, bbox.y1, bbox.x0,
										   bbox.y1 });
		}
	} // namespace

	std::string notehead_cat_name(const BBoxD& head_bbox,
								  const std::vector<double>& staff_lines,
								  double staff_line)
	{
		if (staff_lines.empty())
			return "noteheadBlackOnLine";
		const double cy = (head_bbox.y0 + head_bbox.y1) / 2.0;
		const double tol = std::max(1.0, 2.0 * staff_line);
		for (const double y : staff_lines)
			if (std::abs(cy - y) <= tol)
				return "noteheadBlackOnLine";
		return "noteheadBlackInSpace";
	}

	PageIO build_predicted_page(
		const DatasetImage& image,
		const std::vector<PredictedCropResult>& crop_results,
		const std::vector<std::vector<double>>& staff_lines_per_crop,
		double staff_line)
	{
		PageIO page;
		page.image_id = image.id;
		page.filename = image.filename;
		page.width = image.width;
		page.height = image.height;

		std::vector<std::pair<double, std::string>> duration_pairs;
		std::vector<std::pair<double, std::string>> pitch_pairs;
		for (int i = 0; i < static_cast<int>(crop_results.size()); ++i)
		{
			const PredictedCropResult& result = crop_results[i];
			if (result.split == nullptr || result.graph == nullptr)
				throw std::invalid_argument("crop result has null split/graph");
			const std::vector<double> empty;
			const auto& staff_lines =
				i < static_cast<int>(staff_lines_per_crop.size())
				? staff_lines_per_crop[i]
				: empty;

			auto [annotations, id_map] = annotations_from_split(
				*result.split, i, result.origin, staff_lines, staff_line);
			page.annotations.insert(page.annotations.end(), annotations.begin(),
									annotations.end());
			auto edges = edges_from_split(*result.split, id_map);
			page.graph_edges.insert(page.graph_edges.end(), edges.begin(),
									edges.end());
			auto durations =
				duration_sequence_from_graph(*result.graph, result.origin);
			duration_pairs.insert(duration_pairs.end(), durations.begin(),
								  durations.end());
			auto pitches =
				pitch_sequence_from_graph(*result.graph, result.origin);
			pitch_pairs.insert(pitch_pairs.end(), pitches.begin(),
							   pitches.end());
		}
		std::sort(duration_pairs.begin(), duration_pairs.end());
		for (const auto& [x, token] : duration_pairs)
		{
			(void)x;
			page.duration_sequence.push_back(token);
		}
		std::sort(pitch_pairs.begin(), pitch_pairs.end());
		for (const auto& [x, token] : pitch_pairs)
		{
			(void)x;
			page.pitch_sequence.push_back(token);
		}
		return page;
	}

	PageIO derive_ground_truth(const DeepScoresDataset& dataset,
							   const std::string& image_id,
							   std::optional<double> staff_space,
							   std::function<bool(const BBoxD&)> bbox_filter,
							   const std::vector<double>& staff_lines)
	{
		const DatasetImage& image = dataset.images.at(image_id);
		PageIO page;
		page.image_id = image.id;
		page.filename = image.filename;
		page.width = image.width;
		page.height = image.height;

		std::unordered_set<std::string> scoped_cat_ids;
		for (const auto& name : SCOPED_CATEGORY_NAMES)
		{
			if (auto it = dataset.category_ids.find(name);
				it != dataset.category_ids.end())
				scoped_cat_ids.insert(it->second);
		}

		struct GT
		{
			std::string aid;
			const DatasetAnnotation* ann = nullptr;
			std::string cat_name;
		};
		std::vector<GT> gt;
		if (auto it = dataset.by_image.find(image_id);
			it != dataset.by_image.end())
		{
			for (const auto& aid : it->second)
			{
				const auto& ann = dataset.annotations.at(aid);
				if (ann.cat_id.empty() || !scoped_cat_ids.count(ann.cat_id[0]))
					continue;
				if (bbox_filter && !bbox_filter(ann.a_bbox))
					continue;
				gt.push_back(
					{ aid, &ann, dataset.categories.at(ann.cat_id[0]).name });
			}
		}

		std::vector<BBoxD> bboxes;
		std::vector<std::string> ann_ids;
		std::vector<std::string> cat_names;
		for (const auto& item : gt)
		{
			page.annotations.push_back(Annotation{ item.aid, item.ann->a_bbox,
												   item.cat_name,
												   item.ann->area, 1.0 });
			bboxes.push_back(item.ann->a_bbox);
			ann_ids.push_back(item.aid);
			cat_names.push_back(item.cat_name);
		}

		if (!staff_space)
		{
			std::vector<double> head_heights;
			for (std::size_t i = 0; i < cat_names.size(); ++i)
			{
				if (cat_names[i] == "noteheadBlackOnLine"
					|| cat_names[i] == "noteheadBlackInSpace")
					head_heights.push_back(bboxes[i].y1 - bboxes[i].y0);
			}
			staff_space = head_heights.empty()
				? 4.0
				: std::max(4.0, median(head_heights));
		}
		const double link_d = std::max(2.0, 0.35 * *staff_space);

		std::vector<int> stem_idx;
		std::vector<int> beam_idx;
		for (int i = 0; i < static_cast<int>(cat_names.size()); ++i)
		{
			if (cat_names[i] == "stem")
				stem_idx.push_back(i);
			else if (cat_names[i] == "beam")
				beam_idx.push_back(i);
		}

		std::vector<std::pair<double, std::string>> duration_tokens;
		std::vector<std::pair<double, std::string>> pitch_tokens;
		for (int i = 0; i < static_cast<int>(cat_names.size()); ++i)
		{
			if (cat_names[i] != "noteheadBlackOnLine"
				&& cat_names[i] != "noteheadBlackInSpace")
				continue;
			if (stem_idx.empty())
				continue;

			const int s = *std::min_element(
				stem_idx.begin(), stem_idx.end(), [&](int lhs, int rhs) {
					return bbox_distance(bboxes[i], bboxes[lhs])
						< bbox_distance(bboxes[i], bboxes[rhs]);
				});
			if (bbox_distance(bboxes[i], bboxes[s]) > link_d)
				continue;
			page.graph_edges.push_back(
				{ "head_to_stem", ann_ids[i], ann_ids[s] });
			std::string token = "quarter";
			if (!beam_idx.empty())
			{
				const int b = *std::min_element(
					beam_idx.begin(), beam_idx.end(), [&](int lhs, int rhs) {
						return bbox_distance(bboxes[s], bboxes[lhs])
							< bbox_distance(bboxes[s], bboxes[rhs]);
					});
				if (bbox_distance(bboxes[s], bboxes[b]) <= link_d)
				{
					page.graph_edges.push_back(
						{ "stem_to_beam", ann_ids[s], ann_ids[b] });
					token = "eighth";
				}
			}
			duration_tokens.emplace_back(bboxes[i].x0, token);

			if (!staff_lines.empty())
			{
				const double cy = (bboxes[i].y0 + bboxes[i].y1) / 2.0;
				if (auto ref_y = nearest_system_bottom(staff_lines, cy))
				{
					const int step = static_cast<int>(
						std::lround((*ref_y - cy) / (*staff_space / 2.0)));
					pitch_tokens.emplace_back(bboxes[i].x0,
											  step_to_pitch(step, 'E', 4));
				}
			}
		}

		std::sort(duration_tokens.begin(), duration_tokens.end());
		for (const auto& [x, token] : duration_tokens)
		{
			(void)x;
			page.duration_sequence.push_back(token);
		}
		std::sort(pitch_tokens.begin(), pitch_tokens.end());
		for (const auto& [x, token] : pitch_tokens)
		{
			(void)x;
			page.pitch_sequence.push_back(token);
		}
		return page;
	}

	void write_predictions_json(
		const PageIO& page,
		const std::unordered_map<std::string, DatasetCategory>&
			source_categories,
		const std::filesystem::path& out_path)
	{
		std::unordered_map<std::string, std::string> name_to_id;
		std::set<std::string> scoped_names(std::begin(SCOPED_CATEGORY_NAMES),
										   std::end(SCOPED_CATEGORY_NAMES));
		nlohmann::json scoped_categories = nlohmann::json::object();
		for (const auto& [cid, cat] : source_categories)
		{
			if (cat.annotation_set == "deepscores")
			{
				name_to_id[cat.name] = cid;
				if (scoped_names.count(cat.name))
					scoped_categories[cid] = cat.raw;
			}
		}

		nlohmann::json annotations_json = nlohmann::json::object();
		for (const auto& ann : page.annotations)
		{
			auto it = name_to_id.find(ann.cat_name);
			if (it == name_to_id.end())
				continue;
			annotations_json[ann.ann_id] = {
				{ "a_bbox", bbox_json(ann.bbox) },
				{ "o_bbox", obox_json(ann.bbox) },
				{ "cat_id", nlohmann::json::array({ it->second }) },
				{ "area", ann.area },
				{ "img_id", page.image_id },
				{ "comments", "" },
			};
		}

		nlohmann::json image_id_value = page.image_id;
		if (!page.image_id.empty()
			&& std::all_of(page.image_id.begin(), page.image_id.end(),
						   [](unsigned char c) { return std::isdigit(c); }))
			image_id_value = std::stoll(page.image_id);

		nlohmann::json edge_json = nlohmann::json::array();
		for (const auto& edge : page.graph_edges)
			edge_json.push_back({ { "relation", edge.relation },
								  { "src", edge.src_id },
								  { "dst", edge.dst_id } });

		nlohmann::json payload = {
			{ "info",
			  { { "description", "OMR pipeline predictions" },
				{ "version", "1.0" } } },
			{ "annotation_sets", nlohmann::json::array({ "deepscores" }) },
			{ "categories", scoped_categories },
			{ "images",
			  nlohmann::json::array({ { { "id", image_id_value },
										{ "filename", page.filename },
										{ "width", page.width },
										{ "height", page.height } } }) },
			{ "annotations", annotations_json },
			{ "graph", { { "edges", edge_json } } },
			{ "duration_sequence", page.duration_sequence },
		};

		std::filesystem::create_directories(out_path.parent_path());
		std::ofstream out(out_path);
		if (!out)
			throw std::runtime_error("failed to open " + out_path.string());
		out << payload.dump(2);
	}

} // namespace omr
