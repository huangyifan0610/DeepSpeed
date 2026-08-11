# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from deepspeed import comm as dist
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

    def __init__(self, timers):
        self.timers = timers
        self.event_counters = {}

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

    def _log_bandwidth_impl(self, numel: int, msec: float) -> str:
        ELEMENT_SIZE: int = 2
        bytes: int = numel * ELEMENT_SIZE
        giga_bytes: float = float(bytes) / 1024 / 1024 / 1024
        sec: float = msec / 1000
        bandwidth: float = giga_bytes / sec
        return f'bandwidth = {bandwidth} GB/s | numel = {numel} | size = {giga_bytes} GB | time = {msec} ms'

    def _log_bandwidth(self, fwd: bool = False, bwd: bool = False):
        # forward_fetch_submit: __class__.EventCounter = self.event_counters['forward_fetch_submit']
        # forward_fetch_wait: __class__.EventCounter = self.event_counters['forward_fetch_wait']
        # forward_prefetch_submit: __class__.EventCounter = self.event_counters['forward_prefetch_submit']
        # backward_fetch_submit: __class__.EventCounter = self.event_counters['backward_fetch_submit']
        # backward_fetch_wait: __class__.EventCounter = self.event_counters['backward_fetch_wait']
        # backward_prefetch_submit: __class__.EventCounter = self.event_counters['backward_prefetch_submit']
        forward_all_gather: __class__.EventCounter | None = self.event_counters.get('forward_all_gather')
        backward_all_gather: __class__.EventCounter | None = self.event_counters.get('backward_all_gather')
        forward_all_gather_numel = 0 if forward_all_gather is None else forward_all_gather.num_elem
        backward_all_gather_numel = 0 if backward_all_gather is None else backward_all_gather.num_elem
        forward_numel = forward_all_gather_numel
        backward_numel = backward_all_gather_numel
        total_numel = forward_numel + backward_numel

        forward_fetch_submit_msec = self.timers('forward_fetch_submit').elapsed(reset=False)
        forward_fetch_wait_msec = self.timers('forward_fetch_wait').elapsed(reset=False)
        forward_prefetch_submit_msec = self.timers('forward_prefetch_submit').elapsed(reset=False)
        backward_fetch_submit_msec = self.timers('backward_fetch_submit').elapsed(reset=False)
        backward_fetch_wait_msec = self.timers('backward_fetch_wait').elapsed(reset=False)
        backward_prefetch_submit_msec = self.timers('backward_prefetch_submit').elapsed(reset=False)
        forward_all_gather_msec = self.timers('forward_all_gather').elapsed(reset=False)
        backward_all_gather_msec = self.timers('backward_all_gather').elapsed(reset=False)
        forward_msec = forward_fetch_submit_msec + forward_fetch_wait_msec + forward_prefetch_submit_msec + forward_all_gather_msec
        backward_msec = backward_fetch_submit_msec + backward_fetch_wait_msec + backward_prefetch_submit_msec + backward_all_gather_msec
        total_msec = forward_msec + backward_msec

        my_rank: int = dist.get_rank() if dist.is_initialized() else -1
        forward_bandwidth = self._log_bandwidth_impl(total_numel, total_msec)
        backward_bandwidth = self._log_bandwidth_impl(total_numel, total_msec)
        total_bandwidth = self._log_bandwidth_impl(total_numel, total_msec)

        message = f'[Rank {my_rank}] {total_bandwidth}';
        if fwd:
            message += f'\n\tforward: {forward_bandwidth}'
        if bwd:
            message += f'\n\tbackward: {backward_bandwidth}'
        print(message)

    def log_events(self):
        self._log_event_counters()
        self._log_timers()
        self._log_bandwidth()
