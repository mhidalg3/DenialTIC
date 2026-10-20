"""Static/offline validations for the fresh-GPU-VM bootstrap package.

These do not substitute for a real GPU/Isaac integration test.
"""
from pathlib import Path
import json
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FreshVMTests(unittest.TestCase):
    def test_bash_scripts_parse(self):
        for name in ("bootstrap_fresh_vm.sh", "build_offline_bundle.sh", "vm.sh"):
            subprocess.run(["bash", "-n", str(ROOT / name)], check=True)

    def test_noninteractive_sudo_without_validation_password_prompt(self):
        s = (ROOT / "bootstrap_fresh_vm.sh").read_text()
        self.assertIn('sudo() { command sudo -n "$@"; }', s)
        self.assertIn('sudo true || fatal', s)
        self.assertNotIn('sudo -v\n', s)
        self.assertNotIn('sudo -n -v', s)

    def test_vm_helpers_never_prompt_for_sudo(self):
        for name in ("vm.sh", "build_offline_bundle.sh"):
            content = (ROOT / name).read_text()
            self.assertIn('sudo() { command sudo -n "$@"; }', content)
            self.assertNotRegex(content, r"(?m)^sudo -v\s*$")
        vm = (ROOT / "vm.sh").read_text()
        self.assertIn("sudo true", vm)
        b = (ROOT / "bootstrap_fresh_vm.sh").read_text()
        self.assertIn('["sudo", "-n", "docker", "inspect"', b)
        self.assertNotIn('["sudo", "docker", "inspect"', b)

    def test_bootstrap_dry_plan(self):
        r = subprocess.run(["bash", str(ROOT / "bootstrap_fresh_vm.sh"), "--plan"],
                           check=True, capture_output=True, text=True)
        self.assertIn("9fa6f8b66b9e121d5df5df071297bba8e5353ebb", r.stdout)
        self.assertIn("nvcr.io/nvidia/isaac-sim:5.0.0", r.stdout)
        self.assertIn("Extern", r.stdout)

    def test_manifest_pins_match_downloader_and_checkpoint_sha(self):
        m = json.loads((ROOT / "manifests/dependencies.json").read_text())
        d = (ROOT / "download_models.py").read_text()
        b = (ROOT / "bootstrap_fresh_vm.sh").read_text()
        self.assertIn(m["source"]["commit"], b)
        self.assertIn(m["base_model"]["revision"], d)
        self.assertIn(m["checkpoint"]["revision"], d)
        self.assertIn(m["checkpoint"]["sha256"], d)
        self.assertIn("snapshot_download", d)
        self.assertIn("hf_hub_download", d)
        self.assertNotIn("snapshot_download(\n            repo_id=CHECKPOINT_REPO", d)

    def test_preserves_native_gpu_driver(self):
        s = (ROOT / "bootstrap_fresh_vm.sh").read_text()
        self.assertIn("command -v nvidia-smi", s)
        self.assertNotIn("nvidia-driver-", s)
        self.assertNotIn("apt-get install -y cuda", s)

    def test_docker_not_assumed_and_no_destructive_container_reset(self):
        s = (ROOT / "bootstrap_fresh_vm.sh").read_text()
        self.assertIn("command -v docker", s)
        self.assertIn("nvidia-ctk runtime configure --runtime=docker", s)
        self.assertNotIn("docker rm -f", s)
        self.assertIn("--entrypoint /bin/bash", s)
        self.assertIn("Configuring released DynaNav benchmark for headless cloud rendering", s)
        self.assertIn('"headless": True', s)
        self.assertIn("source .env.container", (ROOT / "vm.sh").read_text())

    def test_host_and_container_validation_are_separated(self):
        vm = (ROOT / "vm.sh").read_text()
        self.assertIn("Checking fresh-VM bootstrap scripts and installer on host", vm)
        self.assertIn("-p 'test_bootstrap.py'", vm)
        self.assertIn("-p 'test_install.py'", vm)
        self.assertIn("-p test_core.py", vm)
        self.assertIn("-p test_adapter.py", vm)
        self.assertNotIn("unittest discover -s DynaNav/trial4/tests -v", vm)
        installer = (ROOT / "install.py").read_text()
        self.assertIn('for filename in ("test_core.py", "test_adapter.py", "test_suite.py")', installer)
        self.assertNotIn("shutil.copytree(tests, dsttests)", installer)

    def test_git_installed_inside_container_before_validation(self):
        s=(ROOT/"bootstrap_fresh_vm.sh").read_text()
        self.assertIn('docker exec -u 0',s)
        self.assertIn('install -y --no-install-recommends git ca-certificates procps',s)
        self.assertIn('git config --global --add safe.directory /workspace/TIC-VLA',s)
        self.assertLess(s.index('Installing container-side system tools'),s.index('Running Trial 4 setup validators'))
        runner=(ROOT/"payload/run_suite.py").read_text()
        self.assertIn('shutil.which("git")',runner)

    def test_protect_native_pytorch_stack(self):
        s=(ROOT/"bootstrap_fresh_vm.sh").read_text()
        self.assertIn('protect_runtime.py --write',s)
        self.assertIn('--constraint /tmp/ticvla-runtime-constraints.txt',s)
        self.assertIn('protect_runtime.py --verify',s)

    def test_large_offline_zip_includes_scripts_and_runtime(self):
        s = (ROOT / "build_offline_bundle.sh").read_text()
        for x in ("docker commit", "docker save", "models.tar.gz", "repo.tar.gz", "SHA256SUMS", "zip -0 -q -r", "cp -a"):
            self.assertIn(x, s)
        b = (ROOT / "bootstrap_fresh_vm.sh").read_text()
        for x in ("--offline-dir", "sha256sum -c", "docker load", "repo.tar.gz", "models.tar.gz"):
            self.assertIn(x, b)

if __name__ == "__main__":
    unittest.main()
