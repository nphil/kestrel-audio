from kestrel_audio.gpu import GpuInfo, NO_GPU, choose_device

FREE = GpuInfo(True, "Tesla P40", 9000, 24576)


def test_uses_the_gpu_when_enough_memory_is_free():
    assert choose_device(FREE, min_free_mib=2500)[0] == "cuda"


def test_falls_back_to_the_cpu_when_the_gpu_is_busy():
    dev, why = choose_device(GpuInfo(True, "Tesla P40", 2400, 24576), min_free_mib=2500)
    assert dev == "cpu" and "busy" in why
    assert choose_device(GpuInfo(True, "Tesla P40", 2500, 24576), min_free_mib=2500)[0] == "cuda"   # exactly enough is enough


def test_no_gpu_means_cpu():
    assert choose_device(NO_GPU, min_free_mib=2500)[0] == "cpu"
    assert choose_device(GpuInfo(True, None, None, None), min_free_mib=2500)[0] == "cpu"


def test_an_open_breaker_keeps_work_off_the_gpu_for_a_while():
    dev, why = choose_device(FREE, min_free_mib=2500, breaker_open=True)
    assert dev == "cpu" and "failed" in why


def test_configuration_can_force_either_device():
    assert choose_device(FREE, min_free_mib=2500, forced="cpu")[0] == "cpu"
    assert choose_device(GpuInfo(True, "x", 100, 24576), min_free_mib=2500, forced="cuda")[0] == "cuda"
    assert choose_device(FREE, min_free_mib=2500, breaker_open=True, forced="cuda")[0] == "cpu"   # a breaker beats "cuda"


# ---------------------------------------------------------------------------------------------- footprint (who is "us" on a shared GPU)
from kestrel_audio.gpu import Footprint


def table_of(state):
    return lambda: dict(state)


def test_the_footprint_is_the_one_new_process_that_appears_when_our_model_loads():
    gpu = {100: 9000, 200: 3000}                      # llama-swap and a neighbour are already there
    fp = Footprint(table_of(gpu))
    assert fp.mib() is None                           # nothing claimed yet
    gpu[777] = 640                                    # our CUDA context
    assert fp.claim() == 777 and fp.mib() == 640
    gpu[777] = 1010
    assert fp.mib() == 1010                           # follows our own process


def test_programs_that_start_after_us_are_not_counted_as_ours():
    gpu = {100: 9000}
    fp = Footprint(table_of(gpu))
    gpu[777] = 640
    fp.claim()
    gpu[888] = 12000                                  # llama-swap loads a big model while a job runs
    assert fp.mib() == 640


def test_when_two_new_processes_appear_together_we_say_unknown_instead_of_guessing():
    gpu = {100: 9000}
    fp = Footprint(table_of(gpu))
    gpu[777] = 640
    gpu[888] = 12000                                  # a neighbour started in the same few seconds
    assert fp.claim() is None and fp.mib() is None
    assert fp.claim() is None                         # claiming is once-only: no later guess either


def test_the_footprint_is_unknown_when_our_process_has_left_or_the_driver_cannot_say():
    gpu = {100: 9000}
    fp = Footprint(table_of(gpu))
    gpu[777] = 640
    fp.claim()
    del gpu[777]
    assert fp.mib() is None
    assert Footprint(table_of({})).claim() is None
