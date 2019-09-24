from cx_Freeze import setup,Executable

base=None
executables = [Executable("final project sample.py",base=base)]

packages=["idna","os","sys"]
options={
    'build.exe': {
        'packages':packages,
    },
}       
    
setup(name="road edge",
      version="",
      description="",
      executables=[Executable(r"opencv_image_sharpening.py")]
      )
