# Encode video for target file size, maintaining quality
When encoding video, encoder options specify a target bitrate or a target quality (quantizer scaled 
compression).  If you choose a bitrate that is too low, the video is overcompressed. If the bitrate 
is too high, the file's size is wasteful.  Setting the quantizer value solves both these problems,
but the output file size is unpredictable.

This script takes an input file or section of video, a quantizer, and a target file size, and makes
encoding attempts while changing the output resolution until the file is just under the target file 
size.  Leveraging scikit-learn's 
[GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
, it efficiently guesses resolutions to minimize encode time.

# Features
* clip a video by specifying the start and end timestamp
* crop a video
* output in WEBM and MP4 containers
* output in VP9 codec for WEBM
* output in H.264 (x264) or AV1 (libaom-av1) for MP4
* burn in subtitles
* direct copy audio when the output is MP4 ffmpeg supports it, otherwise transcode to OPUS
* resync audio (positive float delays audio relative to video, negative the opposite)
* batch output several clips from a single file
* resume encoding guesswork if a job was canceled (incompatible with batches)

# How to use
See [snipnclip.example.yml](snipnclip.example.yml) for an example configuration file.  Rename it to 
`snipnclip.example.yml` for the script to automatically find it, or execute the script with the
configuration yml as the first and only argument to run it.

## Initial configuration
Set the values for `working directory` to an empty directory.  If the script encodes a video down to
the `low resolution limit` and it is still not under the `target size`, the script skips that job.
When using `resume: yes`, be sure to delete any incompletely encoded video from the `working directory`.

## Configuration for encoding
Set the `input`, `container` and `target directory`, then define jobs in a list of YAML objects under
the `snips` configuration key.

```yaml
 - start: "10:36.052"
   end: "11:00.326"
   name: mayo
   crop:
    x: 0
    y: 138
    width: 1920
    height: 804
```

Only the `name` key is required.  If `start` is omitted, it is implied to be the start of the input file. 
Likewise for `end`.  If `crop` is included, all four dimensions are required.  They can be retrieved from 
an image editor like GIMP.

### Timestamps
Timestamps may be an integer, a floating point number, or a string.  If they are a string, they should be
in the form `HH:MM:SS.sss` where `HH`, `MM`, and `.sss` are optional.
