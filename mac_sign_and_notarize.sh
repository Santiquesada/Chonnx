# Sign and notarize the Onnx chump package, for macos
# This is required as Apple by default doesn't let you run
# code that hasn't been built yourself without going through
# this process.

# where the PlinkyRev build can be found
CHUGIN=./Chonnx.chug

# build everything
make clean
make mac

# codesign
echo "codesigning..."
codesign --deep --force --verify --verbose --timestamp --options runtime --entitlements Chugin.entitlements --sign "Developer ID Application" ${CHUGIN}

# build package
chuck -s build-pkg-mac.ck

# where the PlinkyRev build can be found
CHUMP_ZIP=./Chonnx_mac.zip

echo "notarizing..."

echo xcrun notarytool submit ${CHUMP_ZIP} \
    --key "$NOTARIZATION_KEY" \
    --key-id "$NOTARIZATION_KEY_ID" \
    --issuer "$NOTARIZATION_ISSUER" \
    --wait

xcrun notarytool submit ${CHUMP_ZIP} \
    --key "$NOTARIZATION_KEY" \
    --key-id "$NOTARIZATION_KEY_ID" \
    --issuer "$NOTARIZATION_ISSUER" \
    --wait
