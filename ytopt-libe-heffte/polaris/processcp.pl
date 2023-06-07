#!/usr/bin/perl
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# processcp.pl: copy the file processexe.pl to change tne proper number of nodes
#

print "Use 'processexe$ARGV[0].pl' as 'processexe.pl'\n";
system("cp processexe$ARGV[0].pl processexe.pl");
system("chmod +x processexe.pl");
exit 0;

