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
	    print STDERR  "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 1 --ppn 1 --depth $depth --cpu-bind depth sh \$filename > \$tmpname 2>&1\");\n";
	    print OUTFILE "    system(\"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n 1 --ppn 1 --depth $depth --cpu-bind depth sh \$filename > \$tmpname 2>&1\");", "\n";
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
