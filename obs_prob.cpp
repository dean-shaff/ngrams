#include <iostream> 
#include <vector> 
#include <stdio.h>
#include <fstream>
#include <string>
using namespace std ;

void print_vec(vector<int> vec){
	for (int i=0; i<vec.size(); i++){
		cout << vec[i] << ", "; 
	}
	cout << endl ;
}
void print_vec(vector<char> vec){
	for (int i=0; i<vec.size(); i++){
		cout << vec[i] << ", "; 
	}
	cout << endl ;
}

vector<int> slice(vector<int> vec, int start_index, int end_index){
	vector<int> new_vec(end_index-start_index);
	for (int i=start_index; i<end_index; i++){
		new_vec[i-start_index] = vec[i];
	}
	return new_vec;
}

vector<char> slice(vector<char> vec, int start_index, int end_index){
	vector<char> new_vec(end_index-start_index);
	for (int i=start_index; i<end_index; i++){
		new_vec[i-start_index] = vec[i];
	}
	return new_vec;
}

bool compare_array(vector<char> first, vector<char> second){ 
	bool returnval = false;
	if (first.size() != second.size()){
		printf("Vectors are not the same size!\n");
	}
	else{ 
		for (int i=0; i<first.size(); i++){
			if (first.at(i) == second.at(i)){
				returnval = true;
			}else{
				returnval = false ;
			}
		}
	}
	return returnval;
}

double obs_prob(vector<char> biglist, vector<char> indices) {
	double prob ; 
	int n = indices.size();
	int n_big = biglist.size();
	vector<int> reduced ;
	for (int i=0; i<n_big; i++){
		if (indices.at(0) == biglist.at(i)){
			reduced.push_back(i) ; 
		}
	}
	prob = (double) reduced.size() / (double) n_big ;
	printf("Initial Probability: %f\n", prob);;
	if (prob == 0.0){
		return prob;
	}
	for (int i=1; i < n; i++){
		char target = indices[i];
		vector<char> given = slice(indices, 0, i);
		int total_occ = 0 ; 
		int target_occ = 0 ;
		vector<int> new_reduced ;
		for (int j=0; j<reduced.size(); j++){
			int index = reduced[j];
			vector<char> biglist_slice = slice(biglist,index,index+i);
			if (compare_array(biglist_slice, given) == true){
				total_occ += 1 ; 
				new_reduced.push_back(j) ;
				if (biglist[i+index] == target){
					target_occ += 1;
				}
			}
		}
		// reduced.swap(new_reduced);
		// int reduced_size = reduced.size();
		// int new_reduced_size = new_reduced.size();
		// printf("Target: %c \n", target);
		// print_vec(given);
		// printf("Reduced size: %i \n",reduced_size);
		// printf("New Reduced size: %i \n",new_reduced_size);
		// printf("%i, %i\n",total_occ, target_occ);
		if (target_occ == 0 || total_occ == 0){
			return 0.0 ; 
		}
		prob = prob * ((double)target_occ / (double)total_occ);	
	}	
	return prob; 
}

vector<char> fillVector(char *fromarray, int size){
	vector<char> tofill (size) ; 
	for (int i=0 ; i < size; i++){
		tofill.at(i) = fromarray[i]; 
	}
	return tofill ; 
}

vector<int> fillVector(int *fromarray, int size){
	vector<int> tofill (size) ; 
	for (int i=0 ; i < size; i++){
		tofill.at(i) = fromarray[i]; 
	}
	return tofill ; 
}


vector<char> readFile(char *filename){

	ifstream inFile(filename, ios::binary);
	// below is some magic I found on the web.
	std::vector<char> fileContents((istreambuf_iterator<char>(inFile)),
                               istreambuf_iterator<char>());
	inFile.close();
	return fileContents;
}

int main() { 

	char filename[] = "texts/AustenPride.txt";
	vector<char> fileContents = readFile(filename);

	char example[3] = {'t','h','e'}; // this is pointer to first element in array.
	int range[10] = {0,1,2,3,4,5,6,7,8,9};
	// vector<char> filled ('t','h','e'); 
	double val ;
	vector<char> the = fillVector(example, 3) ;
	for (int i=0; i<10; i++){
		val = obs_prob(fileContents, the);
	}
	
	// vector<int> test = fillVector(range,10);
	// vector<int> sliced = slice(test,0,3);
	// print_vec(sliced);
	// printf("hey\n");
	printf("Probability: %f \n", val);
	printf("Done\n");
	return 0;
}