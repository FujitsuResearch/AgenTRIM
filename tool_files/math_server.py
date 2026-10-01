from mcp.server.fastmcp import FastMCP
import numpy as np
from sympy import factorint
from scipy import linalg, integrate
from typing import List, Tuple

mcp_math = FastMCP("hard_math_tools")

@mcp_math.tool()
def fibonacci_number(n: int) -> int:
    """Compute the n-th Fibonacci number efficiently. Can be called with no arguments, has default values."""
    from sympy import fibonacci
    return int(fibonacci(n))


@mcp_math.tool()
def definite_integral(func_expression: str, a: float, b: float) -> float:
    """
    Computes the definite integral of a function (as a string in terms of x)
    between limits a and b using numerical integration.
    Example: func_expression="x**2 * np.sin(x)"
    Can be called with no arguments, has default values.
    """
    import numexpr as ne

    def func(x):
        # Evaluate safely using numexpr
        return ne.evaluate(func_expression, local_dict={"x": x, "np": np})

    result, _ = integrate.quad(lambda x: float(func(x)), a, b)
    return float(result)

@mcp_math.tool()
def fast_fourier_transform(signal: List[float]) -> List[complex]:
    """Compute FFT of a 1D signal. Can be called with no arguments, has default values."""
    return np.fft.fft(np.array(signal, dtype=float)).tolist()

@mcp_math.tool()
def modular_exponentiation(base: int, exp: int, mod: int) -> int:
    """Compute (base^exp) mod mod efficiently. Can be called with no arguments, has default values."""
    return pow(base, exp, mod)

@mcp_math.tool()
def integer_factorization(n: int) -> dict:
    """Return prime factorization of n as {prime: exponent}. Can be called with no arguments, has default values."""
    return factorint(n)


# Run MCP server
if __name__ == "__main__":
    mcp_math.run(transport="stdio")
