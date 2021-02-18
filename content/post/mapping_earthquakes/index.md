---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Retrieving, Analyzing and Visualizing georeferenced data"
subtitle: "Using Folium Map and Standard Python Libraries"
summary: "This post will show you how to map earthquakes from a database using standard Python libraries."
authors: []
tags: []
categories: []
date: 2020-04-25T10:32:45-04:30
lastmod: 2020-04-25T10:32:45-04:30
featured: false
draft: false



# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: True

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []


---

This post contains three [jupyter notebooks](https://jupyter.org/) that will show you how to map earthquakes from a database using standard Python and [Folium](https://python-visualization.github.io/folium/quickstart.html) libraries. The database was filled out from a CSV file obtained from the [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/).


1. <a href="https://nbviewer.jupyter.org/github/acoiman/mapping_earthquakes/blob/master/reading_dataset.ipynb" target="_blank">reading_dataset</a> contains the code to read through the Rdatasers and look for dataset links containing the terms latitude and longitude.


2. <a href="https://nbviewer.jupyter.org/github/acoiman/mapping_earthquakes/blob/master/db_earthquakes.ipynb" target="_blank">db_earthquakes</a> creates a database from the selected dataset and computes some spatial statistics.


3. <a href="https://nbviewer.jupyter.org/github/acoiman/mapping_earthquakes/blob/master/map_earthquakes.ipynb" target="_blank">map_earthquakes</a> takes the database data and creates a web map using the Folium package.

> Click [here](https://towardsdatascience.com/retrieve-analyze-and-visualize-georeferenced-data-aec1af28445b) to read the full post on [Towards Data Science](https://towardsdatascience.com/).


Repository link: <a href="https://github.com/acoiman/mapping_earthquakes" target="_blank">`https://github.com/acoiman/mapping_earthquakes`</a>

***Final result:***

<div class='embed-responsive' style='padding-bottom:75%'>
    <object data='../../maps/earthquake_fiji/' width='100%' height='100%' position: relative display: block height: 0></object>
</div>





