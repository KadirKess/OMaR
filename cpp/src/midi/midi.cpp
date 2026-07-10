#include "midi/midi.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace omr
{
	namespace
	{
		const std::unordered_map<char, int> PITCH_LETTER_OFFSET{
			{ 'C', 0 }, { 'D', 2 }, { 'E', 4 },	 { 'F', 5 },
			{ 'G', 7 }, { 'A', 9 }, { 'B', 11 },
		};

		void write_be16(std::vector<std::uint8_t>& out, std::uint16_t v)
		{
			out.push_back(static_cast<std::uint8_t>(v >> 8));
			out.push_back(static_cast<std::uint8_t>(v & 0xFF));
		}

		void write_be32(std::vector<std::uint8_t>& out, std::uint32_t v)
		{
			out.push_back(static_cast<std::uint8_t>(v >> 24));
			out.push_back(static_cast<std::uint8_t>(v >> 16));
			out.push_back(static_cast<std::uint8_t>(v >> 8));
			out.push_back(static_cast<std::uint8_t>(v & 0xFF));
		}

		void write_varlen(std::vector<std::uint8_t>& out, std::uint32_t v)
		{
			std::uint32_t buffer = v & 0x7F;
			while ((v >>= 7) > 0)
			{
				buffer <<= 8;
				buffer |= ((v & 0x7F) | 0x80);
			}
			while (true)
			{
				out.push_back(static_cast<std::uint8_t>(buffer & 0xFF));
				if (buffer & 0x80)
					buffer >>= 8;
				else
					break;
			}
		}
	} // namespace

	std::optional<int> pitch_token_to_midi(const std::string& token)
	{
		if (token.empty() || token == "?" || token.size() < 2)
			return std::nullopt;
		const char letter = static_cast<char>(
			std::toupper(static_cast<unsigned char>(token[0])));
		auto it = PITCH_LETTER_OFFSET.find(letter);
		if (it == PITCH_LETTER_OFFSET.end())
			return std::nullopt;
		try
		{
			const int octave = std::stoi(token.substr(1));
			return 12 * (octave + 1) + it->second;
		}
		catch (...)
		{
			return std::nullopt;
		}
	}

	std::optional<int> duration_token_to_ticks(const std::string& token,
											   int ppq)
	{
		if (token == "quarter")
			return ppq;
		if (token == "eighth")
			return ppq / 2;
		return std::nullopt;
	}

	int write_midi(const std::vector<std::string>& duration_sequence,
				   const std::vector<std::string>& pitch_sequence,
				   const std::filesystem::path& out_path, int ppq,
				   int tempo_bpm)
	{
		std::vector<std::uint8_t> track;

		// set_tempo meta: delta 0, FF 51 03, 3-byte tempo
		const std::uint32_t tempo =
			60000000u / static_cast<std::uint32_t>(tempo_bpm);
		write_varlen(track, 0);
		track.push_back(0xFF);
		track.push_back(0x51);
		track.push_back(0x03);
		track.push_back(static_cast<std::uint8_t>((tempo >> 16) & 0xFF));
		track.push_back(static_cast<std::uint8_t>((tempo >> 8) & 0xFF));
		track.push_back(static_cast<std::uint8_t>(tempo & 0xFF));

		int written = 0;
		const std::size_t n =
			std::min(duration_sequence.size(), pitch_sequence.size());
		for (std::size_t i = 0; i < n; ++i)
		{
			auto duration = duration_token_to_ticks(duration_sequence[i], ppq);
			if (!duration)
				continue;
			auto note = pitch_token_to_midi(pitch_sequence[i]);
			if (!note)
				continue;

			// note_on: delta 0, status 0x90, note, velocity 64
			write_varlen(track, 0);
			track.push_back(0x90);
			track.push_back(static_cast<std::uint8_t>(*note));
			track.push_back(0x40);

			// note_off: delta duration, status 0x80, note, velocity 0
			write_varlen(track, static_cast<std::uint32_t>(*duration));
			track.push_back(0x80);
			track.push_back(static_cast<std::uint8_t>(*note));
			track.push_back(0x00);
			++written;
		}

		// end of track: delta 0, FF 2F 00
		write_varlen(track, 0);
		track.push_back(0xFF);
		track.push_back(0x2F);
		track.push_back(0x00);

		std::vector<std::uint8_t> smf;
		smf.reserve(14 + 8 + track.size());

		// MThd chunk
		smf.push_back('M');
		smf.push_back('T');
		smf.push_back('h');
		smf.push_back('d');
		write_be32(smf, 6);
		write_be16(smf, 0); // format 0
		write_be16(smf, 1); // 1 track
		write_be16(smf, static_cast<std::uint16_t>(ppq));

		// MTrk chunk
		smf.push_back('M');
		smf.push_back('T');
		smf.push_back('r');
		smf.push_back('k');
		write_be32(smf, static_cast<std::uint32_t>(track.size()));
		smf.insert(smf.end(), track.begin(), track.end());

		std::filesystem::create_directories(out_path.parent_path());
		std::ofstream out(out_path, std::ios::binary);
		if (!out)
			throw std::runtime_error("failed to open " + out_path.string());
		out.write(reinterpret_cast<const char*>(smf.data()),
				  static_cast<std::streamsize>(smf.size()));
		return written;
	}

} // namespace omr
