import unittest
import jax
import jax.numpy as jnp

class TestJax(unittest.TestCase):
    def test_devices_available(self):
        """
        Test that JAX can list at least one available device.
        """
        devices = jax.devices()
        self.assertGreater(len(devices), 0, "No devices found for JAX.")
        print("JAX Devices:", devices)

    def test_gpu_available(self):
        """
        Test that JAX has at least one GPU available.
        If no GPU is found, the test is skipped.
        """
        devices = jax.devices()
        # Use the 'platform' attribute to check for GPU devices.
        gpu_devices = [device for device in devices if device.platform == 'gpu']
        if not gpu_devices:
            self.skipTest("No GPU devices found for JAX; skipping GPU test.")
        self.assertGreater(len(gpu_devices), 0, "No GPU devices found for JAX.")
        print("JAX GPU Devices:", gpu_devices)

    def test_simple_computation(self):
        """
        Run a simple JAX computation (vector addition) and verify the result.
        """
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = a + b
        expected = jnp.array([5, 7, 9])
        # Verify the computation using JAX's allclose.
        self.assertTrue(jnp.allclose(result, expected),
                        f"Simple computation failed: {result} != {expected}")

if __name__ == '__main__':
    unittest.main()
