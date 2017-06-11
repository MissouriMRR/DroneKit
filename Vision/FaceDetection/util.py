#!/usr/bin/env python3.5

def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate