let base64Image = '';
		
$("#image-selector").change(function() {
	let reader = new FileReader();
	
	//after FileReader reads the whole image file		
	reader.onload = function(e){

		//store the image as a Base64 formatted URL string
		let dataURL = reader.result;

		//displaying the selected image
		$('#selected-image').attr("src", dataURL)

		//removing the unnecessary metadata from Base64 string
		base64Image = dataURL.replace("data:image/png;base64,", "").replace("data:image/jpeg;base64,", "").replace("data:image/jpg;base64,", "");
		
		//console.log(base64Image);
	}
	
	//read the content of the selected image file		
	reader.readAsDataURL($("#image-selector")[0].files[0]);
	
	//clearing all the previous text
	$("#prediction").text("");
});
		
$("#predict-button").click(function(event){
	let message = {
		image: base64Image
	}
			
	//console.log(message);
	
	//sending a POST request to our API endpoint		
	$.post("http://127.0.0.1:5000/predict", JSON.stringify(message), function(response){
		
		//prediction results
		pred = response.prediction;
		console.log(response);

		msg = '';

		if(pred == 1)
			msg = 'INFECTED !';
		else
			msg = 'NOT INFECTED.';

		//displaying the prediction
		$("#prediction").text(msg);

		if(pred == 1)
			$("#prediction").css("color", "red");
		else
			$("#prediction").css("color", "green");
	});
});