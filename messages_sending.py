#!/usr/bin/python3
# -*- coding: utf-8 -*-

L = ['Hello', 666, 'World', 888, 'IBM', 'Apple']
L1 = [s.lower() for s in L if isinstance(s, str)]
print(L1)