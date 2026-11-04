"""Exercise the actual macOS first-directory branch without audio hardware."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class StandaloneFirstLaunchTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("clang++"), "requires a C++ compiler")
    def test_new_preferences_directory_uses_private_permissions_and_canonical_filename(self):
        source = (ROOT / "iPlug2/IPlug/APP/IPlugAPP_host.cpp").read_text()
        function = source.split("bool IPlugAPPHost::InitState()", 1)[1].split(
            "void IPlugAPPHost::UpdateINI", 1)[0]
        branch = function.split("#elif defined OS_MAC", 2)[2].split("#else", 1)[0]
        harness = r'''
#include <sys/stat.h>
#include <string>
#include <fstream>
struct Text {
  std::string value;
  const char* Get() const { return value.c_str(); }
  void Append(const char* text) { value += text; }
};
struct Host {
  Text mINIPath;
  void UpdateINI() { std::ofstream(mINIPath.Get()) << "[audio]\nsr=44100\n"; }
  bool create() {
BRANCH
    return true;
  }
};
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  Host host; host.mINIPath.value = std::string(argv[1]) + "/";
  return host.create() ? 0 : 1;
}
'''.replace("BRANCH", branch)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpp = root / "first_launch.cpp"
            cpp.write_text(harness)
            executable = root / "first_launch"
            subprocess.run(["clang++", "-std=c++17", str(cpp), "-o", str(executable)], check=True)
            prefs = root / "DessAmp"
            subprocess.run([str(executable), str(prefs)], check=True)
            self.assertEqual([p.name for p in prefs.iterdir()], ["settings.ini"])
            self.assertIn("sr=44100", (prefs / "settings.ini").read_text())
            self.assertEqual(prefs.stat().st_mode & 0o777, 0o700)
        self.assertNotIn("umask(", branch)


if __name__ == "__main__":
    unittest.main()
