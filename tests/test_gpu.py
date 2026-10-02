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
