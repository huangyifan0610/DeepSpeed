# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from deepspeed.utils import log_dist


class PartitionedParameterProfiler(object):

    @dataclass
    class EventCounter:
        name: str
        count: int
        num_elem: int

        def reset(self):
            self.count = 0
            self.num_elem = 0

        def increment(self, numel):
            self.count += 1
            self.num_elem += numel

    @dataclass
    class BandwidthCounter:
        element_size: int
        numel: int
        msec: float

    def __init__(self, timers):
        self.timers = timers
        self.event_counters = {}
        self.bandwidth_counter = __class__.BandwidthCounter(
            element_size=2,
            numel=0,
            msec=0.0,
        )

    def reset_events(self):
        for event_ctr in self.event_counters.values():
            event_ctr.reset()

    def start_event(self, name):
        if self.timers is None:
            return

        if name not in self.event_counters:
            self.event_counters[name] = __class__.EventCounter(name=name, count=0, num_elem=0)
        self.timers(name).start()

    def stop_event(self, name, num_elem):
        if self.timers is None:
            return
        assert name in self.event_counters, f'unknown event {name}'
        self.event_counters[name].increment(num_elem)
        self.timers(name).stop()
        self.bandwidth_counter.numel += num_elem
        self.bandwidth_counter.msec += self.timers(name).elapsed(reset=False)

    def _log_timers(self):
        if self.timers is None:
            return
        self.timers.log(names=list(self.event_counters.keys()))

    def _log_event_counters(self):
        for event_ctr in self.event_counters.values():
            log_dist(
                f'{event_ctr.name}: count = {event_ctr.count}, numel = {event_ctr.num_elem}',
                #f'{event_ctr.name}: time = {self._log_timers()},count = {event_ctr.count}, numel = {event_ctr.num_elem}',
                ranks=[0])

    def log_events(self):
        self._log_event_counters()
        self._log_timers()

    def log_bandwidth(self, reset: bool=True):
        bytes: int = self.bandwidth_counter.element_size * self.bandwidth_counter.numel
        giga_bytes: float = float(bytes) / (1024 ** 3)
        sec: float = self.bandwidth_counter.msec / 1000
        bandwidth: float = giga_bytes / sec
        return f'bandwidth = {bandwidth} GB/s | numel = {self.bandwidth_counter.numel} | size = {giga_bytes} GB | time = {self.bandwidth_counter.msec} ms'
