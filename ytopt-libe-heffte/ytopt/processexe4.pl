#!/usr/bin/perl
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# processexe.pl: process the file exe.pl to change tne proper number of nodes
#

$A_FILE = "tmpfile.txt";

$filename1 =  $ARGV[0];
#print "Start to process ", $filename, "...\n";
$fname = ">" . $A_FILE;
$i = 0;
open(OUTFILE, $fname);
open (TEMFILE, $filename1);
while (<TEMFILE>) {
    $line = $_;
    chomp ($line);

    if ($line =~ /system/) {
        if ($ARGV[1] <= 64) {
            $depth = $ARGV[1];
            $j = 1;
        } else { if ($ARGV[1] <= 128) {
            $depth = $ARGV[1]/2;
            $j = 2;
        } else { if ($ARGV[1] <= 192) {
            $depth = $ARGV[1]/3;
            $j = 3;
        } else {
            $depth = $ARGV[1]/4;
            $j = 4;
        }
        }
        }
        print OUTFILE "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 4 --ppn 1 --depth $depth sh \$filename > tmpoutfile.txt 2>&1\");", "\n";
    } else {
        print OUTFILE $line, "\n";
    }
}
close(TEMFILE);
close(OUTFILE);
system("mv $A_FILE $filename1");
system("chmod +x $filename1");
#exit main
exit 0;
