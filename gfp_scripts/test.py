import jax
import jaxlib
print("JAX:", jax.__version__)
print("JAXLIB:", jaxlib.__version__)
print(jax.lib.xla_bridge.get_backend().platform)
print(jax.lib.xla_bridge.get_backend().platform_version)
