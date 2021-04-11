# Piano Sound Synthesis Using Finite Difference Schemes

This repository contains supplementary material for the undergraduate project “Piano Sound Synthesis using Finite Difference Schemes” by Eloise Lardet, in partial fulfilment of the MMath degree.

This includes the Python files, sound files, and animations for all the numerical models listed in Appendix C of the report.


## Animations
Short animated clips of the string for the various models.
The code was modified and used animation.FuncAnimation from matplotlib to create the animations.
An example code for animation purposes is provided for the 1D Wave.

## Code
Python files from Appendix C of the report.

C.1-1dwave 			               : 1D Wave

C.2-stiff_string		          : Stiff String

C.3-stiff_string_damping	    : Stiff String with Damping (implicit and explicit schemes)

C.4-hammer_1dwave		          : 1D Wave with Hammer

C.5-hammer_ss_implicit_damped	: Stiff String with Hammer and Damping

## Sounds
Sound output in the form of .wav files for all the codes listed above.

The ‘Different notes’ folder contains sound files for all ‘C’ notes on the piano from C2 up to C7 using the final model from Appendix C.5. 
The Python file full_model-notes contains all the final model and parameter values to create these notes.

The 'mp3' folder contains the .wav sound files converted to an .mp3 format in case your device is unable to play .wav files.


