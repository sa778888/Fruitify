import React from "react";

const Button = ({text,outlined}) => {
  return (
    <button
      className={`px-6 py-3 rounded-3xl font-semibold transition-all duration-300
            ${
              outlined
                ? "cursor-pointer border border-white text-white bg-transparent hover:bg-white hover:text-black"
                : "cursor-pointer bg-gradient-to-r from-green-500 to-green-700 text-white hover:from-green-700 hover:to-green-500"
            }`}
    >
      {text}
    </button>
  );
};

export default Button;