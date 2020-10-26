from subprocess import run

print("hello")
output = run("pwd", capture_output=True).stdout
print(output)