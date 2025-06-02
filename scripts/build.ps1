Write-Output "BUILDING CUDABOX"
uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation
Write-Output "BUILD COMPLETE"
ls dist

Write-Output "INSTALLING CUDABOX"
pip install (Get-ChildItem ./dist/cudabox*.whl).FullName --force-reinstall
Write-Output "INSTALL COMPLETE"
