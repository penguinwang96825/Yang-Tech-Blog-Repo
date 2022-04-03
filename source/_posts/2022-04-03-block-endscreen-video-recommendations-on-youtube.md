---
title: Block Endscreen Video Recommendations on YouTube
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-03 09:12:54
img: /images/wallhaven-dpz67j.jpg
coverImg: /images/wallhaven-dpz67j.jpg
summary: Video recommendations may appear after a video's playback ends or is about to end on YouTube; these recommendations are based on the viewing history and display thumbnails of videos chosen by YouTube's recommendation algorithm. The most critical issue with these is that they may appear while the video is still playing. They overlay part of the screen and cause a bad experience for the viewer. If you don't want to see those YouTube recommendations, you may configure your content blocker to do so.
categories: Script
tags:
	- YouTube
	- Command Line
---

# Introduction

Video recommendations may appear after a video's playback ends or is about to end on YouTube; these recommendations are based on the viewing history and display thumbnails of videos chosen by YouTube's recommendation algorithm. The most critical issue with these is that they may appear while the video is still playing. They overlay part of the screen and cause a bad experience for the viewer. If you don't want to see those YouTube recommendations, you may configure your content blocker to do so.

# Steps

1. Activate the AdBlock icon in the browser's address bar.

<figure>
  <img src="adblock.png" width=100%>
</figure>

2. Locate the settings icon and activate it to open the preferences.

3. Switch to the my filters tab when the dashboard opens.

4. Add the lines below to the set of rules and save it.

```
##.videowall-endscreen
youtube.com##.html5-endscreen-content
youtube.com##.html5-endscreen
youtube.com##.ytp-ce-element
www.youtube.com##.ytp-ce-element
```

# References

1. https://www.mrguarder.com/2018/10/Youtube-Overlay-Block.html
2. https://influrry.tw/block-youtube-end-screens-overlays/