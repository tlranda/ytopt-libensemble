#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpoutfile.txt";
foreach $filename (@ARGV) {
   #print "Start to preprocess ", $filename, "...\n";
   $ssum = 0.0;
   $nmax = 1;
   @nn = (1..$nmax);
   for(@nn) {
    #$retval = gettimeofday( ); 
    $tmpname = $filename;
    $tmpname =~ s/.sh/.log/;
    system("srun -N 4 -n 4  --ntasks-per-gpu=1  --gpus-per-node=1 --gpu-bind=closest -c 8 --cpu-bind=threads sh $filename > $tmpname 2>&1");
    open (TEMFILE, '<', $tmpname);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Performance:/) {
                ($v3, $v4, $v5) = split(' ', $line);
        }
   }
   if ($v4 == 0 ) {
        printf("-1");
   }
   else {
	printf("%.6f", -1 * $v4);
   }
   close(TEMFILE);
    #$tt = gettimeofday( );
    #$ttotal = $tt - $retval;
    #$ssum = $ssum + $ttotal;
   }
   #$avg = $ssum / $nmax;
 #  print "End to preprocess ", $avg, "...\n";
 #printf("%.3f", $avg);
}
