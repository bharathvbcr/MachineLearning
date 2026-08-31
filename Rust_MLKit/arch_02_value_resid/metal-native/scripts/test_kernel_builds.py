#!/usr/bin/env python3
"""Exercise real build.rs file publication with a fake compiler, not GPU code.

The compiler fixture writes its output in place, as a linker may. Assertions
check held file descriptors and failed publication, independently of shader math.
All builds and files are isolated in temporary directories.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
CRATES = [ROOT / "crates/metal-runtime", ROOT / "arch_02_value_resid/metal-native",
          ROOT / "gemma-metal"]


class BuildPublication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metal-build-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        fixture = (f"#!{sys.executable}\n" + '''import os, sys
from pathlib import Path
name=Path(sys.argv[0]).name
if name == "xcrun":
    print(Path(sys.argv[0]).parent/sys.argv[-1] if "-f" in sys.argv else "/tmp")
else:
    target=Path(sys.argv[sys.argv.index("-o")+1])
    target.write_bytes(os.environ.get("FIXTURE_CONTENT","new-library").encode())
    if name == "metallib" and os.environ.get("FIXTURE_FAIL_LINK") == "1":
        sys.exit(1)
''')
        for name in ["xcrun", "metal", "metallib"]:
            path = self.tools / name
            path.write_text(fixture)
            path.chmod(0o700)

    def setup_crate(self, source):
        root = self.root / source.name
        root.mkdir()
        shutil.copytree(source / "kernels", root / "kernels")
        out = root / "out"
        out.mkdir()
        binary = root / "build-script"
        subprocess.run(["rustc", "--edition=2021", str(source / "build.rs"), "-o", str(binary)],
                       check=True, capture_output=True, timeout=60)
        env = os.environ.copy()
        for key in ["METAL_RUNTIME_SKIP_AOT", "METAL_NATIVE_SKIP_AOT", "GEMMA_METAL_SKIP_AOT"]:
            env.pop(key, None)
        env.update(PATH=str(self.tools)+os.pathsep+env["PATH"], DEVELOPER_DIR="/tmp",
                   CARGO_MANIFEST_DIR=str(root), OUT_DIR=str(out))
        return root, binary, env

    def run_build(self, binary, env):
        return subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=60)

    def test_rebuild_preserves_open_libraries(self):
        for source in CRATES:
            with self.subTest(crate=source.name):
                root, binary, env = self.setup_crate(source)
                first = self.run_build(binary, env)
                self.assertEqual(first.returncode, 0, first.stderr)
                path = Path(re.search(r"cargo:rustc-env=\w+_METALLIB=(.+)", first.stdout)[1])
                with path.open("rb") as pinned, (root/"default.metallib").open("rb") as offline:
                    env["FIXTURE_CONTENT"] = "different-next-library"
                    second = self.run_build(binary, env)
                    self.assertEqual(second.returncode, 0, second.stderr)
                    self.assertEqual(pinned.read(), b"new-library", "rebuild mutated a live library inode")
                    self.assertEqual(offline.read(), b"new-library", "offline copy mutated a live inode")
                    next_path = Path(re.search(r"cargo:rustc-env=\w+_METALLIB=(.+)", second.stdout)[1])
                    self.assertNotEqual(path, next_path, "builds must bake independent immutable paths")
                    self.assertEqual(next_path.read_bytes(), b"different-next-library")

    def test_failed_link_does_not_publish(self):
        for source in CRATES:
            with self.subTest(crate=source.name):
                root, binary, env = self.setup_crate(source)
                first = self.run_build(binary, env)
                self.assertEqual(first.returncode, 0, first.stderr)
                path = Path(re.search(r"cargo:rustc-env=\w+_METALLIB=(.+)", first.stdout)[1])
                env.update(FIXTURE_FAIL_LINK="1", FIXTURE_CONTENT="partial-corrupt-library")
                failed = self.run_build(binary, env)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(path.read_bytes(), b"new-library", "failed linker corrupted prior output")
                self.assertEqual((root/"default.metallib").read_bytes(), b"new-library")

    def test_offline_copy_failure_is_not_silenced(self):
        for source in CRATES:
            with self.subTest(crate=source.name):
                root, binary, env = self.setup_crate(source)
                (root/"default.metallib").mkdir()
                result = self.run_build(binary, env)
                self.assertNotEqual(result.returncode, 0, "offline artifact publication error was ignored")


if __name__ == "__main__":
    unittest.main()
