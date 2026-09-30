@echo off
REM Isolates the green pillarbox bars seen on export.
REM The segment filter graph from export.rs produces BLACK bars under libx264 (verified in a Linux
REM sandbox). This runs the SAME graph through three encoders/variants so we can see which one
REM turns them green. Writes four PNGs into cache\green-diag\ -- just look at them.
setlocal
set "ROOT=%~dp0.."
set "OUT=%ROOT%\cache\green-diag"
if not exist "%OUT%" mkdir "%OUT%"

if not "%~1"=="" goto :real
echo Building a synthetic 768x512 test source...
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=s=768x512:r=24:d=2" -c:v libx264 -pix_fmt yuv420p "%OUT%\src.mp4"
goto :probe
:real
echo Using YOUR clip: %~1
copy /y "%~1" "%OUT%\src.mp4" >nul
:probe
echo.
echo === SOURCE properties (this is what I need to see) ===
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,pix_fmt,color_range,color_space,color_primaries,color_transfer,sample_aspect_ratio,display_aspect_ratio,field_order,nb_frames,r_frame_rate -of default=nw=1 "%OUT%\src.mp4"
echo.

set "FG=[0:v:0]trim=start=0:end=2,setpts=(PTS-STARTPTS)/1.000000,scale=1920:1080:force_original_aspect_ratio=decrease,scale=iw*1.000000:ih*1.000000,eq=brightness=0.000000:contrast=1.000000:saturation=1.000000:gamma=1.000000,hue=h=0.000000,format=rgba,rotate=0.00000000:c=none:ow=rotw(iw):oh=roth(ih),colorchannelmixer=aa=1.000000,setsar=1[fg]"
set "TAIL=[bg][fg]overlay=x=960.000-overlay_w/2:y=540.000-overlay_h/2:shortest=1,fps=24,tpad=stop_mode=clone:stop_duration=0.100,trim=duration=2.000000,setpts=PTS-STARTPTS,format=yuv420p[vout]"
set "BG=color=c=black:s=1920x1080:r=24:d=2.000000[bg]"
set "BGFIX=color=c=black:s=1920x1080:r=24:d=2.000000,format=yuv420p[bg]"

echo [1/3] current graph + libx264 ...
ffmpeg -hide_banner -loglevel error -y -i "%OUT%\src.mp4" -filter_complex "%FG%;%BG%;%TAIL%" -map "[vout]" -an -c:v libx264 -b:v 20M -t 2 "%OUT%\a-x264.mp4"

echo [2/3] current graph + h264_nvenc ...
ffmpeg -hide_banner -loglevel error -y -i "%OUT%\src.mp4" -filter_complex "%FG%;%BG%;%TAIL%" -map "[vout]" -an -c:v h264_nvenc -b:v 20M -t 2 "%OUT%\b-nvenc.mp4"

echo [3/3] bg pinned to yuv420p + h264_nvenc ...
ffmpeg -hide_banner -loglevel error -y -i "%OUT%\src.mp4" -filter_complex "%FG%;%BGFIX%;%TAIL%" -map "[vout]" -an -c:v h264_nvenc -b:v 20M -t 2 "%OUT%\c-nvenc-bgfix.mp4"

for %%F in (a-x264 b-nvenc c-nvenc-bgfix) do (
  if exist "%OUT%\%%F.mp4" ffmpeg -hide_banner -loglevel error -y -i "%OUT%\%%F.mp4" -vf "crop=120:300:15:390" -frames:v 1 "%OUT%\bar-%%F.png"
)

echo.
echo === pixel format / colour metadata of each ===
for %%F in (a-x264 b-nvenc c-nvenc-bgfix) do (
  if exist "%OUT%\%%F.mp4" (
    echo --- %%F ---
    ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt,color_range,color_space,color_transfer,color_primaries -of default=nw=1 "%OUT%\%%F.mp4"
  )
)
echo.
echo Open the bar-*.png files in %OUT%
echo   bar-a-x264.png       should be BLACK
echo   bar-b-nvenc.png      GREEN here means NVENC is the culprit
echo   bar-c-nvenc-bgfix.png BLACK here means pinning the background format fixes it
echo.
pause
endlocal
