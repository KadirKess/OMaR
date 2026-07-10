#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace omr
{

	std::optional<int> pitch_token_to_midi(const std::string& token);

	std::optional<int> duration_token_to_ticks(const std::string& token,
											   int ppq = 480);

	int write_midi(const std::vector<std::string>& duration_sequence,
				   const std::vector<std::string>& pitch_sequence,
				   const std::filesystem::path& out_path, int ppq = 480,
				   int tempo_bpm = 120);

} // namespace omr
