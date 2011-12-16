#include "Panzer_String_Utilities.hpp"

#include <boost/algorithm/string.hpp>

namespace panzer {
  
  void StringTokenizer(std::vector<std::string>& tokens,
		       const std::string& str,
		       const std::string delimiters,bool trim)
  {
    using std::string;

    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of(delimiters, lastPos);
    
    while (string::npos != pos || string::npos != lastPos) {
      // grab token, trim if desired
      std::string token = str.substr(lastPos,pos-lastPos);
      if(trim)
         boost::trim(token);

      // Found a token, add it to the vector.
      tokens.push_back(token);
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
    }
    
  }
}
