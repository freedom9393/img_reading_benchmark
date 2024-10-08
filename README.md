# Benchmarking of reading images

This repository is aimed to benchmark several ways of reading images from directory. 31,000 images are used for testing.

## How to benchmark?

Install dependencies:

```bash
pip install -r requirements.txt
```

And then:
```bash
python benchmark.py
```

## Results
After benchmarking, I got these results (changed the order for better visualization):

```bash
# Tested with 31,000 images

loop:        24.742122650146484
async:       23.832134246826172
tensorflow:  13.449952840805054
multithread: 7.090402364730835
nvidia dali: 6.944435358047485
```

## Thoughts
Further investigation is needed, cuz `nvidia dali` should have been much faster, I guess? This task is just reading images, **so if we read and resize images or add more tasks, nvidia dali would have been faster**. It's just my assumption tho.
