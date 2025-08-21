"""Official benchmark implementations for long context evaluation."""

from .base import BaseBenchmark

# Official implementations only
from .niah_official import NIAHOfficialBenchmark
from .ruler_official import RULEROfficialBenchmark  
from .longbench_official import LongBenchOfficialBenchmark

# Aliases for compatibility
NIAHBenchmark = NIAHOfficialBenchmark
RULERBenchmark = RULEROfficialBenchmark
LongBench = LongBenchOfficialBenchmark
LongBenchV2 = LongBenchOfficialBenchmark

__all__ = [
    "BaseBenchmark",
    "NIAHOfficialBenchmark",
    "RULEROfficialBenchmark",
    "LongBenchOfficialBenchmark",
    # Compatibility aliases
    "NIAHBenchmark", 
    "RULERBenchmark",
    "LongBench",
    "LongBenchV2",
]