Option Explicit

Dim shell, fso, root, stateDir, logPath, scriptPath, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName)
stateDir = fso.BuildPath(root, ".aiv")
If Not fso.FolderExists(stateDir) Then fso.CreateFolder(stateDir)

logPath = fso.BuildPath(stateDir, "launcher.log")
scriptPath = fso.BuildPath(root, "scripts\setup-and-launch.ps1")
command = "cmd.exe /d /c powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & scriptPath & """ 1>>""" & logPath & """ 2>>&1"

exitCode = shell.Run(command, 0, True)
If exitCode <> 0 Then
  MsgBox "Digital Forge could not start. Diagnostic details were written to:" & vbCrLf & logPath, vbCritical, "Digital Forge"
End If
