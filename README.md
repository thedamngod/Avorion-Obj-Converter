# Avorion to Wavefront obj Converter

This is a small command line tool that transforms a ship from Avorions .xml files to Wavefront .obj files, which can be imported into modeling programs like Blender or 3dsMax.

## Installation:
Install Python 3.5+ and make sure that it is added in your PATH variable. You can get Python for example from the [Anaconda website](https://www.anaconda.com/download/).

Go to the [Releases](https://github.com/tretum/Avorion-Obj-Converter/releases) section and download the archive (.zip/.tar.gz) of the latest release.

Unzip/Extract the files from the archive.

## Usage:
You have to open a command line, like cmd/Powershell on Windows or bash/zsh on Linux.

**To execute the converter** enter: `python AvObjConverter.py`

Then you have to follow the instructions that are output on the command line.

The files that are output by the tool are saved in a new folder called "out", that is located in the same directory as the saved ship files from Avorion.

## Disclaimer:
In recent updates the functionality might be broken. Also there are new blocks which might not be  correctly recognized and converted.
Most notably the Hangar blocks are not correctly converted.

Furthermore, UV-Maps are not or only generated with errors and there is no materials file. **The model will be untextured and just applying a material will probably not work**.
