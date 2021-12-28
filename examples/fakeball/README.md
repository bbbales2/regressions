This is an attempt to fit a model on a simulated basketball-like game.

To do generate, fit, and plot results, navigate to the `examples/fakeball` folder and run:

```bash
python generate.py
rat fakeball.rat shots.csv samples
python plot.py
```

This produces a plot `offense.png`. The closer the estimates are to the red dashed line, the
better.

You can modify the `generate.py` script to generate more or less data.

There is also a script `estimate.py` that does the equivalent of the middle command.