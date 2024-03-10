CC = clang

all:
	mkdir -p bin libs
	${CC} -g -fPIC ./src/node_func.c -I ./src/include/ -shared -o ./libs/libnode.so
	${CC} -g -I ./libs/ -I ./src/include/ -c ./src/main.c -o ./libs/neural_network.o
	${CC} -g -o ./bin/neural_network ./libs/neural_network.o -L ./libs/ -lm
	chmod +x ./bin/*

normal:
	mkdir -p bin
	${CC} -g -o ./bin/feature_scale -lm ./src/feature_scale.c

debug:
	mkdir -p bin libs
	${CC} -D DEBUG -g -fPIC ./src/neural_network/node_func.c -I ./src/include/ -shared -o ./libs/liblrgrades.so
	${CC} -D DEBUG -g -I ./libs/ -I ./src/include/ -c ./src/main.c -o ./libs/neural_network.o
	${CC} -g -o ./bin/neural_network ./libs/neural_network.o -L ./libs/ -lm -l lrgrades -l lrcostfn -l readdata

	chmod +x ./bin/*

timer:
	mkdir -p bin libs
	${CC} -D TIMER -g -fPIC ./src/neural_network/node_func.c -I ./src/include/ -shared -o ./libs/liblrgrades.so
	${CC} -D TIMER -g -o ./bin/feature_scale -lm ./src/feature_scale.c
	${CC} -D TIMER -g -I ./libs/ -I ./src/include/ -c ./src/main.c -o ./libs/neural_network.o
	${CC} -g -o ./bin/neural_network ./libs/neural_network.o -L ./libs/ -lm -l lrgrades -l lrcostfn -l readdata

	chmod +x ./bin/*


release:
	mkdir -p bin libs
	${CC} -fPIC ./src/neural_network/node_func.c -I ./src/include/ -shared -o ./libs/liblrgrades.so
	${CC} -I ./libs/ -I ./src/include/ -c ./src/main.c -o ./libs/neural_network.o
	${CC} -o ./bin/neural_network ./libs/neural_network.o -L ./libs/ -lm -l lrcostfn -l lrgrades -l readdata
	${CC} -o ./bin/feature_scale -lm ./src/feature_scale.c

	chmod +x ./bin/*

static:
	mkdir -p bin
	${CC} -g -fPIC ./src/read_from_data_file.c -I ./src/include/ ./src/neural_network/cost_function.c ./src/neural_network/node_func.c ./src/main.c -lm -o ./bin/neural_network
	${CC} -g -o ./bin/feature_scale -lm ./src/feature_scale.c

	chmod +x ./bin/*

clean:
	rm -rf ./bin
	rm -rf ./libs
