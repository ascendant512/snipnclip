from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from datetime import timedelta
from typing import Union
from itertools import zip_longest, repeat
from argparse import ArgumentParser, FileType
from re import fullmatch

import numpy as np
from yaml import safe_load
from ffmpeg import input as make_stream, probe, output
from ffmpeg._run import Error as FFMpegError
# from scipy.stats import linregress
# from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from datasize import DataSize


@dataclass
class Resolution:
	x: int
	y: int
	real: bool = False
	video_bytes: int = 0
	file_bytes: int = 0

	@cached_property
	def pixels(self) -> int:
		return self.x * self.y

	@cached_property
	def overhead(self) -> int:
		return self.file_bytes - self.video_bytes

	def __repr__(self):
		return f"{self.x}x{self.y}, {self.video_bytes} {'actual' if self.real else 'virtual'} bytes"


@dataclass
class ResolutionMap:
	original_x: int
	original_y: int
	target_size: Union[int, float]
	potato_limit: int

	@property
	def aspect_ratio(self) -> float:
		return self.original_x / self.original_y

	def __post_init__(self):
		self.resolutions: dict[int, Resolution] = {
			0: Resolution(0, 0, True)
		}
		self.last_step = self.resolutions[0]
		granularity = 8
		if self.aspect_ratio > 1:
			for x in range(self.original_x, self.potato_limit - granularity, -granularity):
				if x <= self.potato_limit:
					break
				y = round(x / self.aspect_ratio)
				y = y + y % 2 # encoder requirement
				# maybe just let x264 or libvpx-vp9 handle indivisibility by 8 on its own
				# remainder = y % 8
				# if remainder > 4:
				# 	y += remainder
				# else:
				# 	y -= remainder
				possible_resolution = Resolution(x, y)
				self.resolutions[possible_resolution.pixels] = possible_resolution
		else:
			for y in range(self.original_y, self.potato_limit - granularity, -granularity):
				if y <= self.potato_limit:
					break
				x = round(y * self.aspect_ratio)
				x = x + x % 2
				possible_resolution = Resolution(x, y)
				self.resolutions[possible_resolution.pixels] = possible_resolution
		self.resolutions = dict(sorted(self.resolutions.items()))
		self.pixels_index: list[int] = list(self.resolutions.keys())

	def __getitem__(self, item: Union[int, Resolution]) -> Resolution:
		if type(item) is Resolution:
			return self.resolutions[item.pixels]
		else:
			return self.resolutions[self.pixels_index[item]]

	def find_by_pixels(self, pixels: int) -> Resolution:
		return self.resolutions[pixels]

	def index(self, item: Resolution) -> int:
		return self.pixels_index.index(item.pixels)

	def reals_as_nparray(self, which: str):
		my_map = {
			resolution.pixels: getattr(resolution, which)
			for resolution in filter(lambda r: r.real, self.resolutions.values())
		}
		dict_description = np.dtype((np.int_, 2))
		real_resolutions = np.fromiter(my_map.items(), dtype=dict_description)
		return real_resolutions.transpose()

	def north_star(self, size_bytes: int, overhead: int):
		original_resolution = self[-1]
		original_resolution.real = True
		original_resolution.video_bytes = size_bytes - overhead
		original_resolution.file_bytes = size_bytes

	def navigate_warp(self, real_resolutions):
		# a * e^(bx) + c
		exponential = lambda x, a, b, c: a * np.exp(b * x) + c
		#initial guesses provided by LibreOffice Calc!
		magic, *idk = curve_fit(exponential,
									real_resolutions[0],
									real_resolutions[1],
									(5109216, 9.861e-7, self.last_step.overhead))
		nearest = self.last_step
		for r in self.resolutions.values():
			if not r.real:
				r.file_bytes = exponential(r.pixels, *magic)
			if abs(self.target_size - r.file_bytes) < abs(self.target_size - nearest.file_bytes):
				nearest = r

		return self[nearest]

	def navigate_warp2(self, real_resolutions):
		# bounds provided by Anonymous!
		kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e8))
		gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
		prediction = gaussian_process.fit(real_resolutions[0].reshape(-1, 1), real_resolutions[1].reshape(-1, 1))

		nearest = self.last_step
		for r in self.resolutions.values():
			if not r.real:
				r.file_bytes = prediction.predict([[r.pixels]])[0]
			if abs(self.target_size - r.file_bytes) < abs(self.target_size - nearest.file_bytes):
				nearest = r

		return self[nearest]


	def navigate_quickly(self) -> Resolution:
		# such clever, very accurate
		real_resolutions = self.reals_as_nparray("video_bytes")
		regression = linregress(real_resolutions, alternative="greater")

		nearest = self.last_step
		target_video_size = self.target_size - self.last_step.overhead
		for virtual_resolution in filter(lambda r: not r.real, self.resolutions.values()):
			virtual_video_bytes = virtual_resolution.pixels * regression.slope + regression.intercept
			virtual_resolution.video_bytes = virtual_video_bytes
			virtual_resolution.file_bytes = virtual_video_bytes + self.last_step.overhead

			abs_virtual_distance = abs(target_video_size - virtual_resolution.video_bytes)
			nearest_virtual_distance = abs(target_video_size - nearest.video_bytes)
			if abs_virtual_distance < nearest_virtual_distance:
				nearest = virtual_resolution

		return nearest

	def navigate_slowly(self, lower_bound: Resolution, upper_bound: Resolution):
		lower_index = self.index(lower_bound)
		upper_index = self.index(upper_bound)

		# this is the desired end state of the whole search
		if upper_index - lower_index == 1:
			if self[upper_index].file_bytes > self.target_size:
				return self[lower_index]
			else:
				return self[upper_index]

		# latch on to the closest known resolution to the target size
		lower_distance = abs(self.target_size - lower_bound.file_bytes)
		upper_distance = abs(self.target_size - upper_bound.file_bytes)
		# take a step from that completed resolution towards that size
		if upper_distance >= lower_distance:
			return self[lower_index + 1]
		else:
			return self[upper_index - 1]

	def navigate(self) -> Resolution:
		if self.last_step.file_bytes == self.target_size:
			return self.last_step # just in case...

		# prepare sanity check - will fast navigation recommend an already-tried resolution, or one outside
		# the two closest already-tried resolutions to the target file size?
		real_r = self.reals_as_nparray("file_bytes")
		try:
			lower_bound_i = np.where(real_r[1] < self.target_size)[0][-1]
			lower_bound_p = real_r[0][lower_bound_i]
		except IndexError:
			lower_bound_p = self.pixels_index[0]
		# no chance for IndexError here: the script will have exited already if highest resolution <= target size
		upper_bound_i = np.where(real_r[1] > self.target_size)[0][0]
		upper_bound_p = real_r[0][upper_bound_i]

		guidance = self.navigate_quickly()
		if len(real_r[0]) > 2:
			guidance = self.navigate_warp2(real_r)

		if lower_bound_p < guidance.pixels < upper_bound_p:
			return guidance
		else:
			# guess what? it did!
			lower_bound_r = self.find_by_pixels(lower_bound_p)
			upped_bound_r = self.find_by_pixels(upper_bound_p)
			return self.navigate_slowly(lower_bound_r, upped_bound_r)

	def annotate(self, resolution: Resolution, size_bytes: int, overhead_bytes: int):
		this_resolution = self[resolution]
		this_resolution.real = True
		this_resolution.file_bytes = size_bytes
		this_resolution.video_bytes = size_bytes - overhead_bytes
		if 0 in self.resolutions:
			del(self.resolutions[0])
			del(self.pixels_index[0])
		self.last_step = resolution

	def resume(self, wd: Path):
		for failure in wd.iterdir():
			if failure.is_dir():
				continue
			matched = fullmatch(r"(?P<x>\d+)x(?P<y>\d+)", failure.stem)
			if not matched:
				continue
			x, y = int(matched["x"]), int(matched["y"])
			failure_resolution = Resolution(x, y, True)
			try:
				failure_result = probe(str(failure))
			except FFMpegError:
				print(f"Failed to probe: {failure}, delete any partials when resuming!")
				exit(1)
			encode_bytes = int(failure_result["format"]["size"])
			overhead_bytes = Job.calculate_overhead(failure_result)
			self.annotate(failure_resolution, encode_bytes, overhead_bytes)



@dataclass
class Job:
	in_file: dict
	config: dict
	job_spec: dict

	@staticmethod
	def config_time_format(timestamp: Union[str, float, int]) -> float:
		if type(timestamp) in [float, int]:
			return float(timestamp)
		if "." not in timestamp:
			timestamp = timestamp + ".000"
		if ":" not in timestamp:
			timestamp = "0:" + timestamp
		h_m_s, ms = timestamp.split(".")
		try:
			milliseconds = int(ms)
		except (TypeError, ValueError):
			raise ValueError(f"Invalid time spec: {timestamp}")
		s_m_h_duration = [
			int(part)
			for part in reversed(h_m_s.split(":"))
		]
		zipped_durations = list(zip_longest(s_m_h_duration, repeat(0, 3), fillvalue=0))
		duration = timedelta(
			hours=max(zipped_durations[2]),
			minutes=max(zipped_durations[1]),
			seconds=max(zipped_durations[0]),
			milliseconds=milliseconds
		)
		return duration.total_seconds()

	@staticmethod
	def calculate_overhead(ff_probe: dict) -> int:
		if ff_probe["format"]["format_name"] == "matroska,webm":
			return 0
		whole_size = int(ff_probe["format"]["size"])
		video_bitrate = int(ff_probe["streams"][0]["bit_rate"])
		video_duration = float(ff_probe["streams"][0]["duration"])
		video_size = video_bitrate / 8 * video_duration
		return round(whole_size - video_size)

	def __post_init__(self):
		if "start" in self.job_spec:
			self.start = self.config_time_format(self.job_spec["start"])
		else:
			self.start = "beginning"
		if "end" in self.job_spec:
			converted = self.config_time_format(self.job_spec["end"])
			try:
				self.duration = converted - self.start
			except TypeError:
				#setting an end without a start
				self.duration = converted
		else:
			self.duration = "end"
		video_stream_info = self.in_file["streams"][0]
		width, height = video_stream_info["width"], video_stream_info["height"]
		if "crop" in self.job_spec:
			width, height = self.job_spec["crop"]["width"], self.job_spec["crop"]["height"]
		self.target_size = self.config[self.config["container"]]["target size"]
		self.map = ResolutionMap(width, height, self.target_size, self.config["low resolution limit"])
		if self.config["resume"]:
			self.map.resume(Path(self.config["working directory"]))

		subtitles = self.streams_of_type("subtitle")

		self.flags: dict[str, bool] = {
			"subtitles": self.config["subtitles"],
			"mp4": self.config["container"] == "mp4",
			"webm": self.config["container"] == "webm",
			"audio offset": self.config,
			"crop": "crop" in self.job_spec,
			"has subtitles": bool(subtitles)
		}
		if self.flags["webm"]:
			self.flags["audio"] = False
			self.flags["audio override"] = False
		elif self.flags["mp4"]:
			self.flags["audio"] = self.streams_of_type("audio") > 0
		self.flags["audio resync"] = self.flags["audio"] and self.config["mp4"]["audio resync"] != 0

		if self.flags["subtitles"] and subtitles == 0:
			self.flags["subtitles"] = False
			print("There are no subtitles in this input.")

	def execute(self):
		if len(self.map.reals_as_nparray("file_bytes")[0]) == 1:
			# initial run, which might fall under the file size without work
			encode_result = self.do_encode()
			overhead_bytes = self.calculate_overhead(encode_result)
			encode_bytes = int(encode_result["format"]["size"])
			if encode_bytes <= self.target_size:
				self.present_file(Resolution(self.map.original_x, self.map.original_y))
				return

			# allow setting the slope and intercept from the results first try
			self.map.north_star(encode_bytes, overhead_bytes)
		next_resolution = self.map.navigate()
		while not next_resolution.real:
			encode_result = self.do_encode(next_resolution)
			encode_bytes = int(encode_result["format"]["size"])
			overhead_bytes = self.calculate_overhead(encode_result)
			self.map.annotate(next_resolution, encode_bytes, overhead_bytes)
			next_resolution = self.map.navigate()

		# ffmpeg may have simply not done anything, no need to overwrite the target file in case of an error
		if next_resolution.file_bytes != 0:
			self.present_file(next_resolution)

	def audio_channels(self) -> int:
		audio_streams = list(filter(lambda stream: stream["codec_type"] == "audio", self.in_file["streams"]))
		for stream in audio_streams:
			return stream["channels"]
		return 0

	def streams_of_type(self, t: str) -> int:
		audio_streams = filter(lambda stream: stream["codec_type"] == t, self.in_file["streams"])
		return len(list(audio_streams))

	def do_encode(self, resolution: Resolution = None) -> dict:
		stream_args = {
			"hide_banner": None
		}
		if self.start != "beginning":
			stream_args["ss"] = f"{self.start:.3f}"
		if self.duration != "end":
			stream_args["t"] = f"{self.duration:.3f}"
		if self.flags["subtitles"]:
			stream_args["copyts"] = None
		video_stream = make_stream(self.in_file["format"]["filename"], **stream_args).video
		if self.flags["crop"]:
			video_stream = video_stream.crop(self.job_spec["crop"]["x"],
												self.job_spec["crop"]["y"],
												self.job_spec["crop"]["width"],
												self.job_spec["crop"]["height"])
		if self.flags["audio resync"]:
			stream_args["itsoffset"] = self.config["mp4"]["audio resync"]

		if self.flags["audio"]:
			audio_stream = make_stream(self.in_file["format"]["filename"], **stream_args).audio
			if resolution or self.flags["audio resync"] or self.flags["subtitles"]:
				audio_stream = audio_stream.filter("asetpts", "PTS-STARTPTS")

		if resolution:
			video_stream = video_stream.filter("scale", width=resolution.x, height=resolution.y)
			out_name = f"{resolution.x}x{resolution.y}." + self.config["container"]
		else:
			out_name = f"{self.map.original_x}x{self.map.original_y}." + self.config["container"]
		out_path = Path(self.config["working directory"]) / out_name

		if self.flags["subtitles"]:
			# take a moment to appreciate the library auto-escaping special characters in the file name...
			video_stream = video_stream.filter("subtitles", filename=self.in_file["format"]["filename"])
			video_stream = video_stream.filter("setpts", "PTS-STARTPTS")

		# do not copy subtitles stream into the output file
		if self.flags["has subtitles"]:
			stream_args["sn"] = None

		if self.flags["webm"]:
			codec_args = {
				"codec:v": "libvpx-vp9",
				"crf": self.config["webm"]["crf"],
				"deadline": "best",
				"g": 999,  # kf-max-dist (maximize allowable keyframe distance)
				"row-mt": 1,  # Enable row based multi-threading
				"auto-alt-ref": 6,  # Enable automatic alt reference frames, Values greater than 1 enable multi-layer alternate reference frames
				"frame-parallel": 1,  # Enable frame parallel decodability features.
				"lag-in-frames": 25,  # number of frames to look ahead for frametype and rate control.
				"an": None, # drop audio streams
				"sn": None, # drop subtitle streams
				"pass": 1,
				"passlogfile": (Path(self.config["working directory"]) / "2pass")
			}
			#on Windows, /dev/null should be something else
			first_out = output(video_stream, "/dev/null", format=self.config["container"], video_bitrate=0, **codec_args)
			first_out.run(overwrite_output=True)
			codec_args["pass"] = 2
			second_out = video_stream.output(str(out_path), video_bitrate=0, **codec_args)
			second_out.run()

		if self.flags["mp4"]:
			codec_args = {
				"codec:v": self.config["mp4"].get("codec", "libx264"),
				"movflags": "faststart",  # allow streaming (move metadata about container to the start of the file)
				"sn": None, # drop subtitle streams
				"map_chapters": -1,  # don't pass chapter data through,
				"crf": str(self.config["mp4"]["crf"])
				# "pix_fmt": "yuv420p", #need to set the color space when converting from gif
			}
			match codec_args["codec:v"]:
				case "libx264":
					codec_args["preset"] = "veryslow"
					# "profile:v": "high", #it's the default
				case "libaom-av1":
					codec_args["cpu-used"] = 0, # placebo encode quality? default is 1
					codec_args["row-mt"] = 1 # Enable row based multi-threading
				case _:
					print("codec must be one of: libx264, libaom-av1")
					exit(1)

			if self.flags["audio"]:
				# these all invoke -filter_complex, and so preclude -codec:a copy
				if resolution or self.flags["audio resync"] or self.flags["subtitles"]:
					audio_channels = self.audio_channels()
					if audio_channels > 2:
						codec_args["ac"] = 2
					codec_args["codec:a"] = "libopus" # transcoding audio is not optional when using video filters
					codec_args["b:a"] = 64000 * min(2, audio_channels)

				else:
					codec_args["codec:a"] = "copy"
				streams = [video_stream, audio_stream]
			else:
				streams = [video_stream]
			encode = output(*streams, str(out_path), video_bitrate=0, **codec_args)
			try:
				encode.run()
			except FFMpegError as e:
				print(e)
				exit(1)

		return probe(str(out_path))

	def present_file(self, resolution: Resolution):
		source_name = "{x}x{y}.{ext}".format(x=resolution.x, y=resolution.y, ext=self.config["container"])
		source_path = Path(self.config["working directory"]) / source_name
		destination_name = "{n}.{ext}".format(n=self.job_spec["name"], ext=self.config["container"])
		destination_path = Path(self.config["target directory"])
		source_path.rename(destination_path / destination_name)


def config_load() -> dict:
	my_dir = Path(__file__).parent
	my_config = str(my_dir / "snipnclip.yml")
	my_args = ArgumentParser()
	my_args.add_argument("config", type=FileType(), nargs="?", default=my_config)
	my_ns = my_args.parse_args()
	return safe_load(my_ns.config)


def wd_cleanup(wd: Path):
	for trash in working_dir.iterdir():
		if trash.is_file():
			trash.unlink()


if __name__ == "__main__":
	my_config = config_load()
	ff_info = probe(my_config["input"])
	working_dir = Path(my_config["settings"]["working directory"])
	target_dir = Path(my_config["settings"]["target directory"])
	settings = my_config["settings"]
	for extension in ["webm", "mp4"]:
		if type(settings[extension]["target size"]) is str:
			settings[extension]["target size"] = DataSize(settings[extension]["target size"])
	for configured_job in my_config["snips"]:
		if my_config["settings"]["resume"]:
			if len(my_config["snips"]) != 1:
				print("You can only use 'resume' with 1 job!")
				exit(1)
		else:
			wd_cleanup(working_dir)
		this_job = Job(ff_info, my_config["settings"], configured_job)
		this_job.execute()
