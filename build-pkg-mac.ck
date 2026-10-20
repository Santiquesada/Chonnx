@import "Chumpinate"

// Our package version
"1.0.0" => string version;

// instantiate a Chumpinate package
Package pkg("Chonnx");

// Add our metadata...
"Santiago Quesada" => pkg.authors;

"https://github.com/Santiquesada/Chonnx" => pkg.homepage;
"https://github.com/Santiquesada/Chonnx" => pkg.repository;

"MIT" => pkg.license;
"Generic ONNX Runtime inference for ChucK: load any float32 .onnx model and run it in real time." => pkg.description;

["machine learning", "onnx", "inference"] => pkg.keywords;

// generate a package-definition.json
// This will be stored in "Chumpinate/package.json"
"./" => pkg.generatePackageDefinition;

<<< "Defining version " + version >>>;;

// Now we need to define a specific PackageVersion for test-pkg
PackageVersion ver("Chonnx", version);

"10.2" => ver.apiVersion;

"1.5.4.0" => ver.languageVersionMin;

"mac" => ver.os;
"arm64" => ver.arch;

// The chugin file
ver.addFile("./Chonnx.chug");

// These build files are examples as well
ver.addExampleFile("examples/currulao_gru_predict.ck");
ver.addDepsFile("deps/onnxruntime/lib/libonnxruntime.1.dylib");

// The version path
"chugins/Chonnx/" + ver.version() + "/" + ver.os() + "/Chonnx.zip" => string path;

<<< path >>>;

// wrap up all our files into a zip file, and tell Chumpinate what URL
// this zip file (which will be served via ccrma's servers)
ver.generateVersion("./", "Chonnx_mac", "https://ccrma.stanford.edu/~nshaheed/" + path);

chout <= "Use the following commands to upload the package to CCRMA's servers:" <= IO.newline();
chout <= "ssh nshaheed@ccrma-gate.stanford.edu \"mkdir -p ~/Library/Web/chugins/Chonnx/"
      <= ver.version() <= "/" <= ver.os() <= "\"" <= IO.newline();
chout <= "scp Chonnx_mac.zip nshaheed@ccrma-gate.stanford.edu:~/Library/Web/" <= path <= IO.newline();

// Generate a version definition json file, stores this in "chumpinate/<VerNo>/Chumpinate_mac.json"
ver.generateVersionDefinition("Chonnx_mac", "./" );
