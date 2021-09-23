var socket;
openSocket();

function toggleClassifier(){
    if(document.getElementById('enable-toggle').checked){
        socket.emit('class_toggle', 1);
    }   else{
        socket.emit('class_toggle', 0);
    }
}

function openSocket() {
	// initialize socket
	socket = io.connect('http://' + document.domain + ':' + location.port);
	// event listener for when the connection to the server is established
	socket.on('connect', function() {
		console.log('We are connected!')
	});
}
