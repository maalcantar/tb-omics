import re
from lxml import etree
#Path to hmdb xml file
hmdb_all_metabolites = 'data\met_data\serum_metabolites.xml'
#Path to your output file
output_file = 'data\met_data\cleaned_serum_metabolites.xml'

xml_top_tag = 'database'

#print etree.XML('<metabolite>', xml_declaration=True)

with open(hmdb_all_metabolites, 'r+', errors='ignore') as f, open(output_file, 'w') as output:
	#Add a new root to the xml - you can only have one root
	for line_num, line in enumerate(f):
		#If the line contains an xml header, don't write it to
		#the new file, except the first tag
		if re.search('<\?xml version', line) and line_num == 0:
			#write out the xml declaration and the first tag
			output.write(line)
			output.write('<'+xml_top_tag+'>\n')
		elif re.search('<\?xml version', line):
			pass
		else:
			#keep the indentation the same as the original file
			output.write('  '+line)
			#write out something for user to see it's working.


	#Close the root
	output.write('</'+xml_top_tag+'>')
