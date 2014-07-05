ccminer-cryptonight

A modification of Christian Buchner's &amp; Christian H.'s
ccminer project by tsiv for Cryptonight mining.

July 5th 2014
-------------

Massive improvement to interactivity on Windows, should also further help with TDR issues.
Introducing the --bfactor and --bsleep command line parameters allows for control over
execution of the biggest resource hog of the algorithm. Use bfactor to determine how
many parts the kernel is split into and bsleep to insert a short delay between the kernel
launches. The defaults are no splitting / no sleep for Linux and split into 64 (bfactor 6)
parts / sleep 100 microseconds between launches for Windows. These defaults seem to work
wonders on my 750 Ti on Windows 7, once again you may want to tweak according to your 
environment.

June 30th 2014
--------------

I've keep getting asked for donation addresses, here are some
for wallets that I currently have up. Will set up other wallets
on request, in case you feel like donating but don't hold any
of the currencies I currently have addresses for.

* BTC: 1JHDKp59t1RhHFXsTw2UQpR3F9BBz3R3cs
* DRK: XrHp267JNTVdw5P3dsBpqYfgTpWnzoESPQ
* JPC: Jb9hFeBgakCXvM5u27rTZoYR9j13JGmuc2
* VTC: VwYsZFPb6KMeWuP4voiS9H1kqxcU9kGbsw
* XMR: 42uasNqYPnSaG3TwRtTeVbQ4aRY3n9jY6VXX3mfgerWt4ohDQLVaBPv3cYGKDXasTUVuLvhxetcuS16ynt85czQ48mbSrWX

In other news, I just yanked out the code for other alrogithms.
This is now a cryptonight-only miner.


June 24th 2014
--------------

Initial release, compiles and runs on Linux and Windows. 
Documentation is scarce and probably will be, see README.txt
and ccminer --help for the basics.

Before you read further (and you should), I highly recommend
running Linux. There are some issues with running on Windows
that are 

Do note that the cryptonight kernel on this release is FAT
and SLOW, and pretty much makes your Windows computer
unusable while running. If you plan on running it on your
primary desktop with only a single GPU... Well, just don't 
think you'll be using the computer for anything. I haven't
tested it, but I'd expect it'll be fine if you have multiple
GPUs on the system and you run the miner only on the cards
that don't have a display attached. You'll still have the
TDR issue to deal with though:

The kernel also tends to turn just long enough for Windows 
to think the GPU crashed and trigger the driver timeout 
detection and recovery. This is where the kernel launch 
option (-l) hopefully comes in.

The default launch is 40 tread blocks of 8 threads each. Don't
know why, but at least my 750 Ti seems to like 8 thread blocks
best. 40 blocks of 8 is something that, once again on my 750 ti,
manages to run fast enough to finish before the Windows default
2 second timeout. Basically enables you to run the damn thing
without the driver crashing instantly, which is why I made
it the default. Since I only have that one single 750 Ti
to test on Windows, I haven't got the slightest clue how
it works on other GPUs. Your mileage may vary.

I peaked out my hashrate with 60 blocks of 8 threads, you'll
just have to experiment with it until (if) you find the sweet
spot for your cards. Do keep in mind that cryptonight needs
2 MB of memory for each hash, that would mean about 960 MB
of GPU memory for the 8x60 config. Keep that and the amount
of memory on your card in mind while playing around with 
the numbers.
