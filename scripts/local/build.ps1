Write-Host "BUILDING CUDABOX" -ForegroundColor Green
uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation --config-settings=cmake.build-type="RelWithDebInfo"
Write-Host "BUILD COMPLETE" -ForegroundColor Green
ls dist

Write-Host "INSTALLING CUDABOX" -ForegroundColor Green
pip install (Get-ChildItem ./dist/cudabox*.whl).FullName --force-reinstall
Write-Host "INSTALL COMPLETE" -ForegroundColor Green
