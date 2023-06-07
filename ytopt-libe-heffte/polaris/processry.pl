#!/usr/bin/perl
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# processcp.pl: copy the file processexe.pl to change tne proper number of nodes
#

print "Use 'run_ytopt$ARGV[0].py' as 'run_ytopt.py'\n";
system("cp run_ytopt$ARGV[0].py run_ytopt.py");
exit 0;

