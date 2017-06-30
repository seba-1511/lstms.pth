#!/bin/bash

.PHONY: all


all: dev

dev:
	python test_correctness.py

test: correct speed capacity

correct:
	python test_correctness.py

speed: 
	python test_speed.py

capacity:
	python test_capacity.py
