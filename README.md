# Sound Measure
PyQt based app which can record and process measurements with sound card

## Requirements
Python 3.4+
PyQt5, numpy, matplotlib, pydub, sounddevice, soundfile

## Features
- Modes: single/periodic - used for calibrating sensors/live measuring
- Record mono/stereo sound at 44.1/48 kHz
- Up to 2 measurable frequencies
- External real-time processed data graphing (used with precompiled plotter_v2)
- Separate interface for sensors config management
- Possible deffered measurements
- Raw output: save captured data in txt/wav format to inspect it manually

## Screenshots
#### Main window
![Main window](https://i.imgur.com/vanziKr.png)
#### Config interface
![Config window](https://i.imgur.com/i6zxVTd.jpg)
#### Mesurements progress window
![Mesurements progress window](https://i.imgur.com/gR7VIKS.jpg)
#### Real time graphing
![Real time graphing](https://i.imgur.com/eOs2BDk.jpg)

## Other
This project is in "always beta" state made with sticks and stones. But there`s actually full working beta compiled version that is used for actual scientific measurements.
