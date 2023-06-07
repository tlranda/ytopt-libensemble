#!/usr/bin/perl
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# processexe.pl: process the file exe.pl to change the proper number of nodes
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
                print OUTFILE "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 2 --ppn 2 --depth 1 --cpu-bind depth sh \$filename > tmpoutfile.txt 2>&1\");", "\n";
	    } else { if ($ARGV[1] <= 128) {
                print OUTFILE "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 2 --ppn 2 --depth 1 --cpu-bind depth sh \$filename > tmpoutfile.txt 2>&1\");", "\n";
		} else { if ($ARGV[1] <= 192) {
                print OUTFILE "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 2 --ppn 2 --depth 1 --cpu-bind depth sh \$filename > tmpoutfile.txt 2>&1\");", "\n";
		} else {
                print OUTFILE "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 2 --ppn 2 --depth 1 --cpu-bind depth sh \$filename > tmpoutfile.txt 2>&1\");", "\n";
	        }
	       }
	      }
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
