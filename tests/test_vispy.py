import unittest
from vispy import app

# Try to import PyOpenGL to access OpenGL functions.
try:
    from OpenGL import GL
except ImportError:
    GL = None

class TestVispyContext(unittest.TestCase):
    def setUp(self):
        """
        Create a Vispy canvas without displaying it and force the creation
        of the native widget and its OpenGL context.
        """
        self.canvas = app.Canvas(keys='interactive', show=False)
        # Force the creation of the native widget (and therefore the GL context)
        self.canvas.create_native()
        # If available, activate the context using 'activate'
        if hasattr(self.canvas.context, 'activate'):
            self.canvas.context.activate()
        # Otherwise, assume the context is current

    def tearDown(self):
        """Close the canvas to free up resources."""
        self.canvas.close()

    def test_opengl_version(self):
        """
        Test if the OpenGL context created by Vispy is working and if the
        OpenGL version is at least 3.3.
        """
        if GL is None:
            self.skipTest("PyOpenGL is not installed; skipping test.")

        # Retrieve the OpenGL version string.
        version_bytes = GL.glGetString(GL.GL_VERSION)
        self.assertIsNotNone(version_bytes, "Failed to retrieve the OpenGL version string.")

        # Decode the version string (if necessary) and print it for debugging.
        version_str = version_bytes.decode('utf-8') if isinstance(version_bytes, bytes) else version_bytes
        print("OpenGL version:", version_str)

        # Extract the numeric part of the version.
        try:
            # Example version string: "3.3.0 NVIDIA 450.80.02"
            version_number = version_str.split()[0]  # gets "3.3.0"
            major_str, minor_str, *_ = version_number.split('.')
            major = int(major_str)
            minor = int(minor_str)
        except Exception as e:
            self.fail(f"Failed to parse the OpenGL version ('{version_str}'): {e}")

        # Check if the OpenGL version is at least 3.3.
        self.assertTrue((major, minor) >= (3, 3),
                        f"OpenGL version ({major}.{minor}) is lower than the required 3.3.")

if __name__ == '__main__':
    unittest.main()
