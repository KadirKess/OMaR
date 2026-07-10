#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "midi/midi.h"

namespace
{
	std::vector<std::uint8_t> read_file(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary);
		in.seekg(0, std::ios::end);
		const auto size = static_cast<std::streamsize>(in.tellg());
		in.seekg(0, std::ios::beg);
		std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
		in.read(reinterpret_cast<char*>(data.data()), size);
		return data;
	}
} // namespace

TEST(PitchTokenToMidi, BasicNotes)
{
	EXPECT_EQ(omr::pitch_token_to_midi("A4"), 69);
	EXPECT_EQ(omr::pitch_token_to_midi("C4"), 60);
	EXPECT_EQ(omr::pitch_token_to_midi("C-1"), 0);
}

TEST(PitchTokenToMidi, UnknownReturnsNullopt)
{
	EXPECT_FALSE(omr::pitch_token_to_midi("?"));
	EXPECT_FALSE(omr::pitch_token_to_midi(""));
	EXPECT_FALSE(omr::pitch_token_to_midi("A"));
	EXPECT_FALSE(omr::pitch_token_to_midi("X4"));
	EXPECT_FALSE(omr::pitch_token_to_midi("Axx"));
}

TEST(DurationTokenToTicks, KnownTokens)
{
	EXPECT_EQ(*omr::duration_token_to_ticks("quarter", 480), 480);
	EXPECT_EQ(*omr::duration_token_to_ticks("eighth", 480), 240);
	EXPECT_EQ(*omr::duration_token_to_ticks("quarter", 960), 960);
	EXPECT_EQ(*omr::duration_token_to_ticks("eighth", 960), 480);
}

TEST(DurationTokenToTicks, UnknownReturnsNullopt)
{
	EXPECT_FALSE(omr::duration_token_to_ticks("half"));
	EXPECT_FALSE(omr::duration_token_to_ticks("?"));
}

TEST(WriteMidi, WritesValidSmf0)
{
	const auto dir = std::filesystem::temp_directory_path() / "omr_midi_test";
	std::filesystem::remove_all(dir);
	const auto path = dir / "test.mid";

	std::vector<std::string> durations = { "quarter", "eighth", "eighth" };
	std::vector<std::string> pitches = { "A4", "B4", "C5" };

	int written = omr::write_midi(durations, pitches, path, 480, 120);
	EXPECT_EQ(written, 3);

	auto bytes = read_file(path);
	ASSERT_GT(bytes.size(), 14u + 8u);

	EXPECT_EQ(bytes[0], 'M');
	EXPECT_EQ(bytes[1], 'T');
	EXPECT_EQ(bytes[2], 'h');
	EXPECT_EQ(bytes[3], 'd');
	EXPECT_EQ(bytes[4], 0);
	EXPECT_EQ(bytes[5], 0);
	EXPECT_EQ(bytes[6], 0);
	EXPECT_EQ(bytes[7], 6);
	EXPECT_EQ(bytes[8], 0);
	EXPECT_EQ(bytes[9], 0);
	EXPECT_EQ(bytes[10], 0);
	EXPECT_EQ(bytes[11], 1);
	EXPECT_EQ(bytes[12], 0x01);
	EXPECT_EQ(bytes[13], 0xE0);

	EXPECT_EQ(bytes[14], 'M');
	EXPECT_EQ(bytes[15], 'T');
	EXPECT_EQ(bytes[16], 'r');
	EXPECT_EQ(bytes[17], 'k');

	// byte 22 = varlen delta 0, byte 23 = tempo meta start
	EXPECT_EQ(bytes[22], 0x00);
	const auto track_start = 23u;
	EXPECT_EQ(bytes[track_start + 0], 0xFF);
	EXPECT_EQ(bytes[track_start + 1], 0x51);
	EXPECT_EQ(bytes[track_start + 2], 0x03);
	EXPECT_EQ(bytes[track_start + 3], 0x07);
	EXPECT_EQ(bytes[track_start + 4], 0xA1);
	EXPECT_EQ(bytes[track_start + 5], 0x20);

	std::filesystem::remove_all(dir);
}

TEST(WriteMidi, SkipsUnknownPitch)
{
	const auto dir = std::filesystem::temp_directory_path() / "omr_midi_test2";
	std::filesystem::remove_all(dir);
	const auto path = dir / "skip.mid";

	std::vector<std::string> durations = { "quarter", "eighth" };
	std::vector<std::string> pitches = { "?", "C5" };

	int written = omr::write_midi(durations, pitches, path);
	EXPECT_EQ(written, 1);

	std::filesystem::remove_all(dir);
}
