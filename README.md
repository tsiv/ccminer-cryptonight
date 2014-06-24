ccminer

Christian Buchner's &amp; Christian H.'s CUDA miner project

Modified by tsiv to include support for cryptonight mining

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
