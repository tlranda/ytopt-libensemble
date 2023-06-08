#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpoutfile.txt";
$depth = $ARGV[0];
foreach $filename (@ARGV[1 .. $#ARGV]) {
   #print "Start to preprocess ", $filename, "...\n";
   $ssum = 0.0;
   $nmax = 1;
   @nn = (1..$nmax);
   for(@nn) {
    #$retval = gettimeofday( ); 
    $N_NODES = 2;
    print "/opt/cray/pe/pals/1.1.7/bin/mpiexec -n $N_NODES --ppn 4 --depth $depth sh $filename > tmpoutfile.txt 2>&1\n";
    system("/opt/cray/pe/pals/1.1.7/bin/mpiexec -n $N_NODES --ppn 4 --depth $depth sh $filename > tmpoutfile.txt 2>&1");
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Performance:/) {
                ($v3, $v4, $v5) = split(' ', $line);
 		printf("%.6f", -1 * $v4);
        }
   }
   if ($v4 == 0 ) {
        printf("1");
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
